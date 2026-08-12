// Extracts module specifiers from source text so the package-load contract can assert
// that `src/` only imports things Pi's loader provides. No JavaScript/TypeScript parser
// API is a declared dependency of this package (`@typescript/native-preview` ships a
// binary, Biome ships a CLI), so this is a lexical scan hardened in two ways: comments
// and their contents are removed before scanning, and a dynamic import or require whose
// argument is not a plain string literal is reported as UNVERIFIABLE_SPECIFIER rather
// than silently ignored.
export const UNVERIFIABLE_SPECIFIER = "<unverifiable-dynamic-specifier>";

// Replaces comment bodies with spaces, preserving offsets. String and template literals
// are kept, because import specifiers live inside them, but a literal is only treated as
// a specifier when an import token immediately precedes it -- so an ordinary message
// string that happens to contain `from "pkg"` is not read as an import.
function stripCommentsAndLiterals(source: string): string {
  let out = "";
  let index = 0;

  while (index < source.length) {
    const two = source.slice(index, index + 2);

    if (two === "//") {
      const end = source.indexOf("\n", index);
      const stop = end === -1 ? source.length : end;
      out += " ".repeat(stop - index);
      index = stop;
      continue;
    }

    if (two === "/*") {
      const end = source.indexOf("*/", index + 2);
      const stop = end === -1 ? source.length : end + 2;
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
      // Keep the literal intact: import specifiers live inside these quotes.
      out += source.slice(index, stop);
      index = stop;
      continue;
    }

    out += source[index];
    index += 1;
  }

  return out;
}

// `from "x"` and `import "x"` -- static forms, always a plain literal. The import token
// must not itself sit inside a literal, which is what distinguishes a real import from a
// message string quoting one.
const STATIC_SPECIFIER = /(?:\bfrom\s*|\bimport\s+)(["'`])([^"'`]*)\1/g;
// The call forms, whose argument may be an arbitrary expression.
const CALL_START = /\b(?:import|require)\s*\(/g;

// A call argument counts as verifiable only if it is exactly one string literal, or a
// template literal with no substitution. Anything else -- an identifier, a
// concatenation, `${...}` -- cannot be resolved by reading the source.
function literalArgument(argument: string): string | undefined {
  const trimmed = argument.trim();
  const quoted = /^(["'])([^"']*)\1$/.exec(trimmed);
  if (quoted) return quoted[2];
  const template = /^`([^`]*)`$/.exec(trimmed);
  if (template && !template[1].includes("${")) return template[1];
  return undefined;
}

// Offsets of every character that sits inside a string or template literal body.
function literalBodyOffsets(source: string): Set<number> {
  const inside = new Set<number>();
  let index = 0;
  while (index < source.length) {
    const quote = source[index];
    if (quote !== '"' && quote !== "'" && quote !== "`") {
      index += 1;
      continue;
    }
    let cursor = index + 1;
    while (cursor < source.length) {
      if (source[cursor] === "\\") {
        cursor += 2;
        continue;
      }
      if (source[cursor] === quote) break;
      inside.add(cursor);
      cursor += 1;
    }
    index = Math.min(cursor + 1, source.length);
  }
  return inside;
}

export function importSpecifiers(source: string): string[] {
  const scanned = stripCommentsAndLiterals(source);
  const insideLiteral = literalBodyOffsets(scanned);
  const specifiers = [...scanned.matchAll(STATIC_SPECIFIER)]
    .filter((match) => !insideLiteral.has(match.index ?? 0))
    .map((match) => match[2]);

  for (const call of scanned.matchAll(CALL_START)) {
    if (insideLiteral.has(call.index ?? 0)) continue;
    const open = (call.index ?? 0) + call[0].length;
    const close = scanned.indexOf(")", open);
    const argument = scanned.slice(open, close === -1 ? scanned.length : close);
    const literal = literalArgument(argument);
    specifiers.push(literal ?? UNVERIFIABLE_SPECIFIER);
  }

  return specifiers;
}
