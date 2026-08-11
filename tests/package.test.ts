import { execFile } from "node:child_process";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { importSpecifiers } from "./import-specifiers.js";

const execFileAsync = promisify(execFile);
const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

interface PackageManifest {
  main?: string;
  types?: string;
  exports?: unknown;
  files: string[];
  pi: { extensions: string[]; image: string };
}

interface LoadResult {
  errors: unknown[];
  extensions: unknown[];
}

async function expectedPackageFiles(): Promise<string[]> {
  const sourceFiles = (await readdir(join(repoRoot, "src")))
    .filter((file) => file.endsWith(".ts"))
    .map((file) => `package/src/${file}`);
  return ["package/LICENSE", "package/README.md", "package/package.json", ...sourceFiles].sort();
}

async function loadExtension(entrypoint: string, cwd: string): Promise<LoadResult> {
  const loaderUrl = pathToFileURL(
    resolve(repoRoot, "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"),
  ).href;
  const { loadExtensions } = (await import(loaderUrl)) as {
    loadExtensions(paths: string[], cwd: string): Promise<LoadResult>;
  };

  return loadExtensions([entrypoint], cwd);
}

describe("package gallery metadata", () => {
  it("uses the gallery image URL expected by pi.dev", async () => {
    const { default: manifest } = await import("../package.json", {
      with: { type: "json" },
    });

    expect(manifest.pi.image).toBe(
      "https://raw.githubusercontent.com/balcsida/pi-provider-litellm/refs/heads/main/assets/pi_litellm_gallery.png",
    );
  });

  it("does not expose the npm badge as gallery media", async () => {
    const readme = await readFile("README.md", "utf8");

    expect(readme).not.toContain("https://img.shields.io/npm/v/pi-provider-litellm.svg");
  });
});

describe("pi package compatibility", () => {
  it("declares a single source-only Pi entrypoint", async () => {
    const manifest = JSON.parse(await readFile(join(repoRoot, "package.json"), "utf8")) as PackageManifest;

    expect(manifest.pi.extensions).toEqual(["./src/index.ts"]);
    expect(manifest.files).toEqual(["src", "README.md", "LICENSE"]);
    expect(manifest).not.toHaveProperty("main");
    expect(manifest).not.toHaveProperty("types");
    expect(manifest).not.toHaveProperty("exports");
  });

  it("loads a production-only Git archive from its manifest entrypoint", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-git-package-"));
    try {
      const archivePath = join(fixture, "git-package.tar");
      const { stdout: worktreeCommit } = await execFileAsync("git", ["stash", "create"], { cwd: repoRoot });
      const revision = worktreeCommit.trim() || "HEAD";
      await execFileAsync("git", ["archive", "--format=tar", `--output=${archivePath}`, revision], { cwd: repoRoot });
      await execFileAsync("tar", ["-xf", archivePath, "-C", fixture]);
      await rm(archivePath);
      const manifest = JSON.parse(await readFile(join(fixture, "package.json"), "utf8")) as PackageManifest;
      const entrypoint = resolve(fixture, manifest.pi.extensions[0]);

      const result = await loadExtension(entrypoint, fixture);

      expect(result.errors).toEqual([]);
      expect(result.extensions).toHaveLength(1);
    } finally {
      await rm(fixture, { force: true, recursive: true });
    }
  }, 30_000);

  it("packs only source/docs/license and loads the packed manifest entrypoint", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-npm-package-"));
    try {
      const { stdout: tarballName } = await execFileAsync(
        "npm",
        ["pack", "--ignore-scripts", "--pack-destination", fixture],
        { cwd: repoRoot },
      );
      const tarballPath = resolve(fixture, tarballName.trim().split("\n").at(-1) ?? "");
      const { stdout: fileList } = await execFileAsync("tar", ["-tzf", tarballPath]);
      await execFileAsync("tar", ["-xzf", tarballPath, "-C", fixture]);

      const packageRoot = join(fixture, "package");
      const manifest = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8")) as PackageManifest;
      const entrypoint = resolve(packageRoot, manifest.pi.extensions[0]);
      const result = await loadExtension(entrypoint, packageRoot);

      expect(fileList.trim().split("\n").sort()).toEqual(await expectedPackageFiles());
      expect(result.errors).toEqual([]);
      expect(result.extensions).toHaveLength(1);
    } finally {
      await rm(fixture, { force: true, recursive: true });
    }
  }, 30_000);

  it("keeps source runtime imports loader-provided or built-in", async () => {
    const sourceDir = join(repoRoot, "src");
    const sourceFiles = (await readdir(sourceDir)).filter((file) => file.endsWith(".ts"));
    const imports = await Promise.all(
      sourceFiles.map(async (file) => [file, importSpecifiers(await readFile(join(sourceDir, file), "utf8"))] as const),
    );

    for (const [file, specifiers] of imports) {
      expect(specifiers, file).toSatisfy((values: string[]) =>
        values.every(
          (specifier) =>
            specifier.startsWith("node:") ||
            specifier.startsWith("./") ||
            specifier === "@earendil-works/pi-ai" ||
            specifier === "@earendil-works/pi-ai/compat" ||
            specifier === "@earendil-works/pi-ai/providers/all" ||
            specifier === "@earendil-works/pi-coding-agent",
        ),
      );
    }
  });

  it("requires the native Provider extension API", async () => {
    const { default: manifest } = await import("../package.json", {
      with: { type: "json" },
    });

    expect(manifest.peerDependencies["@earendil-works/pi-ai"]).toBe(">=0.81.0");
    expect(manifest.peerDependencies["@earendil-works/pi-coding-agent"]).toBe(">=0.81.0");
    expect(manifest.peerDependenciesMeta).toEqual({
      "@earendil-works/pi-ai": { optional: true },
      "@earendil-works/pi-coding-agent": { optional: true },
    });
    expect(manifest.devDependencies["@earendil-works/pi-ai"]).toBe("^0.83.0");
    expect(manifest.devDependencies["@earendil-works/pi-coding-agent"]).toBe("^0.83.0");
  });

  it("documents native Provider model persistence", async () => {
    const readme = await readFile("README.md", "utf8");

    expect(readme).toContain("Pi 0.81.0+ is required");
    expect(readme).toContain("native Provider");
    expect(readme).toContain("run `/login`, choose `Sign in with an API key`, then choose `LiteLLM API key`");
    expect(readme).toContain("With `/login litellm`, choose `Sign in with an API key` directly");
    expect(readme).toContain("~/.pi/agent/models-store.json");
    expect(readme).toContain("Opening `/model` refreshes configured provider catalogs");
    expect(readme).not.toContain("/litellm-refresh");
    expect(readme).toContain("Legacy `litellm-models*.json` files are ignored and are not deleted");
    expect(readme).not.toContain("older than 24 hours");
    expect(readme).not.toContain("enter `2` for SSO");
  });
});

describe("dependency security overrides", () => {
  it("keeps vulnerable transitive dependencies above alerted ranges", async () => {
    const lockfile = JSON.parse(await readFile("package-lock.json", "utf8")) as {
      packages?: Record<string, { version?: string }>;
    };

    const copiesOf = (name: string): Record<string, string> =>
      Object.fromEntries(
        Object.entries(lockfile.packages ?? {})
          .filter(([path]) => path === `node_modules/${name}` || path.endsWith(`/node_modules/${name}`))
          .map(([path, pkg]) => [path, pkg.version ?? "missing"]),
      );

    // basic-ftp left the dependency tree entirely; its override is vestigial.
    expect(Object.values(copiesOf("basic-ftp")).every((version) => version === "6.0.1")).toBe(true);
    expect(Object.values(copiesOf("brace-expansion"))).toEqual(["5.0.9"]);
    expect(Object.values(copiesOf("undici"))).toEqual(["8.9.0"]);
    // Pi 0.81.1 no longer ships a nested protobufjs copy.
    expect(copiesOf("protobufjs")).toEqual({
      "node_modules/protobufjs": "8.7.1",
    });
  });
});
