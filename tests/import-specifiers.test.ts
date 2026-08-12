import { describe, expect, it } from "vitest";
import { importSpecifiers, UNVERIFIABLE_SPECIFIER } from "./import-specifiers.js";

// The package-load contract in tests/package.test.ts trusts this scanner, and an
// `every()` over an empty array is vacuously true -- so a scanner that silently
// returned nothing would make that contract pass for any source. These cases pin the
// oracle itself.
describe("importSpecifiers", () => {
  it.for([
    ['import "pkg";', ["pkg"]],
    ['import a from "pkg";', ["pkg"]],
    ['import { a } from "pkg";', ["pkg"]],
    ['import\n  { a }\n  from\n  "pkg";', ["pkg"]],
    ['import type { A } from "pkg";', ["pkg"]],
    ['export { a } from "pkg";', ["pkg"]],
    ['export * from "pkg";', ["pkg"]],
    ['export * as ns from "pkg";', ["pkg"]],
    ['const m = require("pkg");', ["pkg"]],
    ['await import("pkg");', ["pkg"]],
    ["await import(`pkg`);", ["pkg"]],
    ['await import(  "pkg" );', ["pkg"]],
  ] as const)("extracts %s", ([source, expected]) => {
    expect(importSpecifiers(source)).toEqual([...expected]);
  });

  it.for([
    ["a line comment", '// import "pkg";\n'],
    ["a block comment", '/* import "pkg"; */'],
    ["a comment mentioning from", '// see from "pkg" for details\n'],
  ] as const)("ignores %s", ([, source]) => {
    expect(importSpecifiers(source)).toEqual([]);
  });

  it.for([
    ["a bare identifier", "await import(name);"],
    // Assembled so the placeholder sequence is test data, not a template in this file.
    ["a template with substitution", `await import(\`$${"{base}"}/mod.js\`);`],
    ["a concatenation", 'await import("pk" + "g");'],
    ["a computed require", "require(names[0]);"],
  ] as const)("reports %s as unverifiable", ([, source]) => {
    expect(importSpecifiers(source)).toContain(UNVERIFIABLE_SPECIFIER);
  });

  it("finds a lazy import inside a function that is never called", () => {
    const source = ["export async function unused() {", '  return await import("pkg");', "}"].join("\n");

    expect(importSpecifiers(source)).toEqual(["pkg"]);
  });

  it.for([
    ["escaped quotes", 'const doc = "import x from \\"pkg\\"";'],
    ["single quotes inside double", `const msg = "resolved from 'pkg'";`],
    ["double quotes inside single", `const msg = 'alias from "pkg" is unknown';`],
    ["a template literal", 'const msg = `alias from "pkg" is unknown`;'],
    ["an error message quoting a dynamic import", `throw new Error("use import('./x.js') instead");`],
  ] as const)("does not read %s as an import", ([, source]) => {
    expect(importSpecifiers(source)).toEqual([]);
  });

  // The sentinel is asserted literally here so renaming the constant cannot silently
  // turn every unverifiable specifier into an allowlisted one.
  it("uses a sentinel no allowlist would accept", () => {
    expect(UNVERIFIABLE_SPECIFIER).toBe("<unverifiable-dynamic-specifier>");
    expect(UNVERIFIABLE_SPECIFIER.startsWith("node:")).toBe(false);
    expect(UNVERIFIABLE_SPECIFIER.startsWith("./")).toBe(false);
  });
});
