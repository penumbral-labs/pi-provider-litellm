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

// Replaces the contents of comments, regexes, and string/template literals with spaces,
// preserving length, so a regex scan sees only executable text.
function blankNonCode(source: string): string {
  let out = "";
  let index = 0;

  while (index < source.length) {
    const two = source.slice(index, index + 2);

    if (two === "//" || two === "/*") {
      const end = two === "//" ? source.indexOf("\n", index) : source.indexOf("*/", index + 2);
      const stop = end === -1 ? source.length : two === "//" ? end : end + 2;
      out += source.slice(index, stop).replace(/[^\n]/g, " ");
      index = stop;
      continue;
    }

    const quote = source[index];
    if (quote === '"' || quote === "'" || quote === "`") {
      let cursor = index + 1;
      while (cursor < source.length) {
        if (source[cursor] === "\\") {
          cursor += 2;
          continue;
        }
        if (source[cursor] === quote) break;
        cursor += 1;
      }
      const stop = Math.min(cursor + 1, source.length);
      out += source.slice(index, stop).replace(/[^\n]/g, " ");
      index = stop;
      continue;
    }

    if (quote === "/" && startsRegexLiteral(source, index)) {
      let cursor = index + 1;
      let inCharacterClass = false;
      while (cursor < source.length) {
        if (source[cursor] === "\\") {
          cursor += 2;
          continue;
        }
        if (source[cursor] === "[") inCharacterClass = true;
        else if (source[cursor] === "]") inCharacterClass = false;
        else if (source[cursor] === "/" && !inCharacterClass) {
          cursor += 1;
          while (/[a-z]/i.test(source[cursor] ?? "")) cursor += 1;
          break;
        }
        cursor += 1;
      }
      out += source.slice(index, cursor).replace(/[^\n]/g, " ");
      index = cursor;
      continue;
    }

    out += source[index];
    index += 1;
  }

  return out;
}

// JavaScript decides whether `/` starts a regex from the preceding token. Resolver calls in
// this package are only rejected, never executed, so this small lexical predicate only needs
// to keep regex contents from changing comment/string state in the policy scan.
function startsRegexLiteral(source: string, slashIndex: number): boolean {
  const prefix = source.slice(0, slashIndex).trimEnd();
  if (prefix === "") return true;

  const previous = prefix.at(-1) ?? "";
  if ("([{:;,=!?&|+-*%^~<>".includes(previous)) return true;

  const word = /[A-Za-z_$][\w$]*$/.exec(prefix)?.[0];
  return word !== undefined && /^(?:await|case|delete|in|instanceof|of|return|throw|typeof|void|yield)$/.test(word);
}

let ready: Promise<void> | undefined;

// es-module-lexer compiles a WASM module on first use.
export async function initImportSpecifiers(): Promise<void> {
  ready ??= init;
  await ready;
}

export function importSpecifiers(source: string, filename = "source.ts"): string[] {
  const [imports] = parse(source, filename);
  const specifiers = imports.map((entry) => entry.n ?? UNVERIFIABLE_SPECIFIER);

  // Blank comment and literal bodies before the resolver scan. Only executable text counts:
  // a doc comment or a message string mentioning `require()` is not a resolver call, and
  // flagging one would fail the package contract for prose.
  const executable = blankNonCode(source);
  for (const pattern of FORBIDDEN_RESOLVERS) {
    if (pattern.test(executable)) specifiers.push(FORBIDDEN_RESOLVER);
  }

  return specifiers;
}
