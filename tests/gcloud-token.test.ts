import { execFile } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CACHE_TTL_MS, getGcloudToken, getGcloudTokenCommand, resetGcloudTokenCache } from "../src/gcloud-token.js";
import { importSpecifiers } from "./import-specifiers.js";

const execFileAsync = promisify(execFile);
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

const ORIGINAL_ENV = {
  APPDATA: process.env.APPDATA,
  GOOGLE_APPLICATION_CREDENTIALS: process.env.GOOGLE_APPLICATION_CREDENTIALS,
  HOME: process.env.HOME,
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function writeAdcFile(body: unknown): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-gcloud-"));
  const path = join(dir, "adc.json");
  await writeFile(path, JSON.stringify(body), "utf8");
  return path;
}

afterEach(() => {
  for (const [key, value] of Object.entries(ORIGINAL_ENV)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  resetGcloudTokenCache();
  vi.restoreAllMocks();
});

describe("getGcloudTokenCommand", () => {
  it("executes a TypeScript token module through Pi's command helper", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-litellm-gcloud-command-"));
    const modulePath = join(fixture, "token-source.ts");
    await writeFile(modulePath, 'export async function getGcloudToken() { return "ya29.subprocess-token"; }\n');

    const command = getGcloudTokenCommand(pathToFileURL(modulePath).href).slice(1);
    const { stdout } = await execFileAsync("/bin/sh", ["-c", command], { cwd: repoRoot });

    expect(stdout).toBe("ya29.subprocess-token");
  });

  it("executes the production ADC module through plain Node", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({ type: "service_account" });
    const command = getGcloudTokenCommand().slice(1);

    await expect(execFileAsync("/bin/sh", ["-c", command], { cwd: repoRoot })).rejects.toMatchObject({
      code: 1,
      stderr: expect.stringContaining("Service account credentials are not supported"),
    });
  });

  it("keeps the plain-Node ADC graph built-in-only and erasable by Node", async () => {
    const cliSource = await readFile(join(repoRoot, "src/gcloud-token-cli.ts"), "utf8");
    const tokenSource = await readFile(join(repoRoot, "src/gcloud-token.ts"), "utf8");

    expect(importSpecifiers(cliSource)).toEqual([]);
    expect(importSpecifiers(tokenSource).every((specifier) => specifier.startsWith("node:"))).toBe(true);
    expect(cliSource).toContain("await import(moduleUrl)");
  });
});

describe("getGcloudToken", () => {
  it("exchanges authorized_user ADC credentials for an access token", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { access_token: "ya29.token" }));

    await expect(getGcloudToken()).resolves.toBe("ya29.token");

    expect(fetchMock).toHaveBeenCalledWith(
      "https://oauth2.googleapis.com/token",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: expect.stringContaining("grant_type=refresh_token"),
      }),
    );
  });

  it("uses the cached token until the TTL expires", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "first-token" }))
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "second-token" }));

    const now = new Date("2026-06-10T12:00:00.000Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(now);
    await expect(getGcloudToken()).resolves.toBe("first-token");
    await expect(getGcloudToken()).resolves.toBe("first-token");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    vi.mocked(Date.now).mockReturnValue(now + CACHE_TTL_MS + 1);
    await expect(getGcloudToken()).resolves.toBe("second-token");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not reuse a cached token after the ADC identity changes", async () => {
    const firstPath = await writeAdcFile({
      type: "authorized_user",
      client_id: "first-client",
      client_secret: "first-secret",
      refresh_token: "first-refresh",
    });
    const secondPath = await writeAdcFile({
      type: "authorized_user",
      client_id: "second-client",
      client_secret: "second-secret",
      refresh_token: "second-refresh",
    });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "first-token" }))
      .mockResolvedValueOnce(jsonResponse(200, { access_token: "second-token" }));
    vi.spyOn(Date, "now").mockReturnValue(new Date("2026-06-10T12:00:00.000Z").getTime());

    process.env.GOOGLE_APPLICATION_CREDENTIALS = firstPath;
    await expect(getGcloudToken()).resolves.toBe("first-token");

    process.env.GOOGLE_APPLICATION_CREDENTIALS = secondPath;
    await expect(getGcloudToken()).resolves.toBe("second-token");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("returns null and warns when no ADC file exists", async () => {
    delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
    delete process.env.APPDATA;
    process.env.HOME = await mkdtemp(join(tmpdir(), "pi-litellm-no-adc-"));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("No Google ADC file found"));
  });

  it("returns null and warns for service account credentials", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({ type: "service_account" });
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("Service account credentials are not supported"));
  });

  it("returns null when the token exchange fails", async () => {
    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile({
      type: "authorized_user",
      client_id: "client-id",
      client_secret: "client-secret",
      refresh_token: "refresh-token",
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("invalid_grant", { status: 400 }));
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await expect(getGcloudToken()).resolves.toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("Token exchange failed"));
  });
});
