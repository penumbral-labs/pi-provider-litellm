import { init, parse } from "es-module-lexer";

// Extracts module specifiers from source text so the package-load contract can assert that
// `src/` only imports things Pi's loader provides. es-module-lexer does the parsing: it is a
// real ES module lexer, so comments, string contents and template literals are handled by
// construction rather than by hand-rolled scanning.
//
// Two things the lexer cannot answer, both handled as policy:
//   - a dynamic `import()` whose specifier is not a plain string literal reports no name,
//     and is surfaced as UNVERIFIABLE_SPECIFIER rather than dropped;
//   - CommonJS and dynamic-evaluation escape hatches are not ES imports at all, so they are
//     rejected by name. `src/` is ESM-only, so any appearance is a contract violation.
export const UNVERIFIABLE_SPECIFIER = "<unverifiable-dynamic-specifier>";
export const FORBIDDEN_RESOLVER = "<forbidden-dynamic-resolver>";

const FORBIDDEN_RESOLVERS = [
  /\brequire\s*\(/,
  /\bcreateRequire\s*\(/,
  /\beval\s*\(/,
  /\bnew\s+Function\s*\(/,
  /\bimport\.meta\.resolve\s*\(/,
];

let ready: Promise<void> | undefined;

// es-module-lexer compiles a WASM module on first use.
export async function initImportSpecifiers(): Promise<void> {
  ready ??= init;
  await ready;
}

export function importSpecifiers(source: string, filename = "source.ts"): string[] {
  const [imports] = parse(source, filename);
  const specifiers = imports.map((entry) => entry.n ?? UNVERIFIABLE_SPECIFIER);

  // Strip comments before the resolver scan so a mention in prose is not a violation. String
  // contents are left alone: a resolver call cannot hide inside a string and still execute.
  const withoutComments = source.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
  for (const pattern of FORBIDDEN_RESOLVERS) {
    if (pattern.test(withoutComments)) specifiers.push(FORBIDDEN_RESOLVER);
  }

  return specifiers;
}
