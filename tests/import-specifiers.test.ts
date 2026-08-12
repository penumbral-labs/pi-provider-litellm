import { beforeAll, describe, expect, it } from "vitest";
import {
  FORBIDDEN_RESOLVER,
  importSpecifiers,
  initImportSpecifiers,
  UNVERIFIABLE_SPECIFIER,
} from "./import-specifiers.js";

// The package-load contract in tests/package.test.ts trusts this oracle, and an assertion
// over an empty result would be vacuously true -- so an oracle returning nothing would make
// that contract pass for any source. These cases pin the oracle itself.
describe("importSpecifiers", () => {
  beforeAll(async () => {
    await initImportSpecifiers();
  });

  it.for([
    ['import "pkg";', ["pkg"]],
    ['import a from "pkg";', ["pkg"]],
    ['import { a } from "pkg";', ["pkg"]],
    ['import\n  { a }\n  from\n  "pkg";', ["pkg"]],
    ['import type { A } from "pkg";', ["pkg"]],
    ['export { a } from "pkg";', ["pkg"]],
    ['export * from "pkg";', ["pkg"]],
    ['export * as ns from "pkg";', ["pkg"]],
    ['await import("pkg");', ["pkg"]],
    ['import a from "./rel.js";', ["./rel.js"]],
    ['import { readFile } from "node:fs/promises";', ["node:fs/promises"]],
  ] as const)("extracts %s", ([source, expected]) => {
    expect(importSpecifiers(source)).toEqual([...expected]);
  });

  it.for([
    ["a line comment", '// import "pkg";\n'],
    ["a block comment", '/* import "pkg"; */'],
    ["a comment mentioning from", '// see from "pkg" for details\n'],
    ["a plain string", `const msg = "resolved from 'pkg'";`],
    ["a string with escaped quotes", 'const doc = "import x from \\"pkg\\"";'],
    ["a template literal", 'const msg = `alias from "pkg" is unknown`;'],
    ["an error message quoting a dynamic import", `throw new Error("use import('./x.js') instead");`],
  ] as const)("ignores %s", ([, source]) => {
    expect(importSpecifiers(source)).toEqual([]);
  });

  it.for([
    ["a bare identifier", "await import(name);"],
    ["a template with substitution", `await import(\`$${"{base}"}/mod.js\`);`],
    ["a plain template literal", "await import(`pkg`);"],
    ["a concatenation", 'await import("pk" + "g");'],
  ] as const)("reports %s as unverifiable", ([, source]) => {
    expect(importSpecifiers(source)).toContain(UNVERIFIABLE_SPECIFIER);
  });

  it.for([
    ["require", 'const m = require("pkg");'],
    ["createRequire", 'const r = createRequire(import.meta.url); r("pkg");'],
    ["eval", 'eval("import(\\"pkg\\")");'],
    ["new Function", 'new Function("return import(\\"pkg\\")");'],
    ["import.meta.resolve", 'import.meta.resolve("pkg");'],
  ] as const)("rejects %s as a forbidden resolver", ([, source]) => {
    expect(importSpecifiers(source)).toContain(FORBIDDEN_RESOLVER);
  });

  it("does not treat a comment mentioning require as a forbidden resolver", () => {
    expect(importSpecifiers("// we never call require() here\n")).toEqual([]);
  });

  it("finds a lazy import inside a function that is never called", () => {
    const source = ["export async function unused() {", '  return await import("pkg");', "}"].join("\n");

    expect(importSpecifiers(source)).toEqual(["pkg"]);
  });

  // Asserted literally so renaming either constant cannot turn a rejected specifier into one
  // an allowlist would silently accept.
  it.for([UNVERIFIABLE_SPECIFIER, FORBIDDEN_RESOLVER] as const)("uses a sentinel no allowlist accepts: %s", (s) => {
    expect(s.startsWith("<")).toBe(true);
    expect(s.startsWith("node:")).toBe(false);
    expect(s.startsWith("./")).toBe(false);
  });
});
