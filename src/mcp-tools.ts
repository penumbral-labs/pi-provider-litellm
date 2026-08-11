import { createHash } from "node:crypto";
import type { Static, TSchema } from "@earendil-works/pi-ai";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionContext, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMMcpTool, LiteLLMRuntimeAuth } from "./types.js";

const LIST_TIMEOUT_MS = 10_000;
const CALL_TIMEOUT_MS = 30_000;
const MAX_DISCOVERY_BODY_BYTES = 5 * 1024 * 1024;
const MAX_CALL_BODY_BYTES = 5 * 1024 * 1024;
const MAX_REGISTERED_TOOLS = 512;
const MAX_DESCRIPTION_BYTES = 4 * 1024;
const MAX_SCHEMA_BYTES = 64 * 1024;
const MAX_SCHEMA_DEPTH = 16;
const MAX_RESULT_BYTES = 64 * 1024;
const MAX_TOOL_NAME_LENGTH = 64;
const TOOL_NAME_HASH_LENGTH = 10;
const MAX_LABEL_BYTES = 256;
const MAX_DETAIL_BYTES = 256;
const MAX_ERROR_WALK_DEPTH = 16;
const MAX_DIAGNOSTIC_SAMPLES = 5;
const MAX_DIAGNOSTIC_CAUSE_BYTES = 200;
const TRUNCATION_MARKER = "\n[truncated by pi-provider-litellm]";
const DESCRIPTION_TRUNCATION_MARKER = "… [truncated]";
const SHORT_TRUNCATION_MARKER = "…";

interface RawLiteLLMMcpTool {
  name?: unknown;
  description?: unknown;
  inputSchema?: unknown;
  input_schema?: unknown;
  server_id?: unknown;
  server_name?: unknown;
  mcp_info?: {
    server_id?: unknown;
    server_name?: unknown;
  };
}

interface PreparedTool {
  tool: LiteLLMMcpTool;
  name: string;
  parameters: TSchema;
  syntheticArgsEnvelope: boolean;
}

/**
 * Why a discovered MCP tool was not registered as the proxy described it.
 * `schema-envelope` is a degradation, not a loss: the tool still registers, with the
 * extension-owned envelope in place of a schema that could not be proven safe.
 */
export type McpDropReason = "invalid-tool" | "duplicate-identity" | "invalid-schema" | "name-collision";
export type McpDegradeReason = "schema-envelope";
export type McpIncidentReason = McpDropReason | McpDegradeReason;

const DROP_REASON_TEXT: Readonly<Record<McpDropReason, string>> = {
  "invalid-tool": "a missing name or server identity",
  "duplicate-identity": "duplicate identities",
  "invalid-schema": "an invalid or oversized input schema",
  "name-collision": "a colliding generated Pi name",
};

const DEGRADE_REASON_TEXT: Readonly<Record<McpDegradeReason, string>> = {
  "schema-envelope": "a schema that could not be proven free of proxy-supplied regexes or references",
};

const INCIDENT_REASON_TEXT: Readonly<Record<McpIncidentReason, string>> = {
  ...DROP_REASON_TEXT,
  ...DEGRADE_REASON_TEXT,
};

export interface McpPreparationReport {
  /** Raw tool entries the proxy returned, before any normalization. */
  discovered: number;
  /** Tools registered with the schema the proxy supplied. */
  prepared: number;
  /** Of `prepared`, how many carry the extension-owned envelope instead of the proxy's schema. */
  enveloped: number;
  /** Tools discarded because the registered-tool cap was reached. */
  overflow: number;
  dropped: Array<{ reason: McpDropReason; tools: string[] }>;
  degraded: Array<{ reason: McpDegradeReason; tools: string[] }>;
}

/** Bounded discovery result: `raw` is what the proxy returned, so losses can be reconciled. */
export interface McpDiscovery {
  raw: number;
  tools: LiteLLMMcpTool[];
  /** Entries dropped by normalization, as bounded safe labels. */
  invalid: string[];
}

// Incident-aware suppression: an incident class is re-reported whenever its message changes
// (different count or different tools), so a stable proxy stays quiet across refreshes while a
// changed one is always surfaced. Keys come from a fixed set of classes, so this cannot grow
// without bound.
const lastEmittedIncident = new Map<string, string>();

/**
 * Reports an incident unless its identity is unchanged since the last report of that class.
 * `identity` must capture everything an operator would want to hear about again — for tool
 * incidents that is the full sorted membership, not just the bounded sample that gets printed.
 */
function emitSafetyDiagnostic(incident: string, message: string, identity: string = message): void {
  if (lastEmittedIncident.get(incident) === identity) return;
  lastEmittedIncident.set(incident, identity);
  process.stderr.write(`LiteLLM MCP: ${message}\n`);
}

function clearIncident(incident: string): void {
  lastEmittedIncident.delete(incident);
}

function membershipIdentity(tools: readonly string[]): string {
  return createHash("sha256")
    .update([...tools].sort().join("\0"))
    .digest("hex")
    .slice(0, 16);
}

function plural(count: number, singular: string, pluralForm = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

/** Renders a bounded sample list. Names are already sanitized to `[a-z0-9_]`, so they are safe for stderr. */
function sampleList(names: readonly string[]): string {
  const shown = names.slice(0, MAX_DIAGNOSTIC_SAMPLES);
  const remainder = names.length - shown.length;
  return remainder > 0 ? `${shown.join(", ")} (+${remainder} more)` : shown.join(", ");
}

/**
 * Reports a registration pass that Pi aborted.
 *
 * Pi's `registerTool` is a synchronous replace-by-name whose only throw comes from a staleness
 * check that is never reset, so a throw means the pass is over, not that one tool was rejected.
 * `cause` is Pi-authored and bounded; no proxy schema, description, or body reaches stderr.
 */
export function reportMcpRegistrationFatal(registered: number, attempted: number, cause: unknown): void {
  const causeText = truncateUtf8(
    cause instanceof Error ? cause.message : String(cause),
    MAX_DIAGNOSTIC_CAUSE_BYTES,
    SHORT_TRUNCATION_MARKER,
  );
  emitSafetyDiagnostic(
    "registration-fatal",
    `registration stopped after ${registered} of ${plural(attempted, "MCP tool")}; ` +
      `no further attempts will be made by this extension instance (${causeText}).`,
  );
}

/** Reports a discovery that yielded no registrable tool, so the silence is explained. */
export function reportMcpEmptyCatalog(raw: number): void {
  emitSafetyDiagnostic(
    "empty-catalog",
    `no MCP tools were registered from ${plural(raw, "raw entry", "raw entries")} returned by the proxy.`,
    `empty-catalog:${raw}`,
  );
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function byteLength(value: string): number {
  return Buffer.byteLength(value, "utf8");
}

function truncateUtf8(value: string, maxBytes: number, marker: string): string {
  if (byteLength(value) <= maxBytes) return value;
  const markerBytes = byteLength(marker);
  const source = Buffer.from(value, "utf8");
  let end = Math.max(0, maxBytes - markerBytes);
  while (end > 0 && (source[end] & 0xc0) === 0x80) end -= 1;
  return `${source.subarray(0, end).toString("utf8")}${marker}`;
}

async function readBoundedText(response: Response, limit: number, surface: string): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        // Build the diagnostic and the error before cancelling: `cancel()` rejects with the stream's
        // stored error if the body errored concurrently, which would otherwise discard both.
        emitSafetyDiagnostic(`${surface}-body-cap`, `${surface} response exceeded its ${limit}-byte limit.`);
        const capError = new Error(`${surface} response exceeds its ${limit}-byte limit`);
        await reader.cancel().catch(() => undefined);
        throw capError;
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  // A response that stayed within the cap clears the incident, so a later breach on this surface
  // is reported again instead of being suppressed as an unchanged message.
  clearIncident(`${surface}-body-cap`);
  return Buffer.concat(
    chunks.map((chunk) => Buffer.from(chunk)),
    size,
  ).toString("utf8");
}

function parseJson(text: string, surface: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`${surface} returned invalid JSON`);
  }
}

function normalizeMcpTool(value: unknown): LiteLLMMcpTool | undefined {
  const raw = asRecord(value) as RawLiteLLMMcpTool | undefined;
  if (!raw) return undefined;
  const name = stringValue(raw.name);
  const serverName =
    stringValue(raw.mcp_info?.server_name) ??
    stringValue(raw.server_name) ??
    stringValue(raw.mcp_info?.server_id) ??
    stringValue(raw.server_id);
  if (!name || !serverName) return undefined;
  const suppliedSchema = raw.inputSchema ?? raw.input_schema;
  const inputSchema = asRecord(raw.inputSchema) ?? asRecord(raw.input_schema);
  return {
    name,
    server_name: serverName,
    server_id: stringValue(raw.mcp_info?.server_id) ?? stringValue(raw.server_id) ?? serverName,
    description: stringValue(raw.description) ?? name,
    input_schema: inputSchema ?? {},
    // Present but not an object: malformed, as distinct from absent.
    ...(inputSchema === undefined && suppliedSchema !== undefined && suppliedSchema !== null
      ? { input_schema_malformed: true }
      : {}),
  };
}

/**
 * A bounded, injection-safe label for an entry that normalization rejected. Such an entry may have
 * no usable name at all, so fall back to a positional placeholder rather than echoing proxy text.
 */
function invalidToolLabel(value: unknown, index: number): string {
  const raw = asRecord(value) as RawLiteLLMMcpTool | undefined;
  const name = raw ? stringValue(raw.name) : undefined;
  const server = raw
    ? (stringValue(raw.mcp_info?.server_name) ?? stringValue(raw.server_name) ?? stringValue(raw.server_id))
    : undefined;
  if (name && server) return `${sanitizeName(server)}_${sanitizeName(name)}`;
  if (name) return `unknown_server_${sanitizeName(name)}`;
  if (server) return `${sanitizeName(server)}_unnamed`;
  return `entry_${index}`;
}

export async function discoverMcpTools(
  baseUrl: string,
  apiKey: string,
  headers?: Record<string, string>,
  onProgress?: (message: string) => void,
  parentSignal?: AbortSignal,
): Promise<McpDiscovery> {
  parentSignal?.throwIfAborted();
  onProgress?.("Discovering MCP tools from server...");
  const signal = parentSignal
    ? AbortSignal.any([parentSignal, AbortSignal.timeout(LIST_TIMEOUT_MS)])
    : AbortSignal.timeout(LIST_TIMEOUT_MS);
  try {
    onProgress?.("Querying MCP tools/list endpoint...");
    const response = await fetch(`${normalizeBaseUrl(baseUrl)}/mcp-rest/tools/list`, {
      headers: {
        ...headers,
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      signal,
    });
    if (!response.ok) {
      onProgress?.(`MCP tools discovery failed with status ${response.status}`);
      throw new Error(`HTTP ${response.status}`);
    }
    const body = parseJson(await readBoundedText(response, MAX_DISCOVERY_BODY_BYTES, "MCP discovery"), "MCP discovery");
    const bodyRecord = asRecord(body);
    const rawTools = Array.isArray(body) ? body : Array.isArray(bodyRecord?.tools) ? bodyRecord.tools : [];
    onProgress?.(`Found ${rawTools.length} raw MCP tools, normalizing...`);
    const tools: LiteLLMMcpTool[] = [];
    const invalid: string[] = [];
    // Every raw entry lands in exactly one bucket, so later counts reconcile back to `raw`.
    rawTools.forEach((raw, index) => {
      const tool = normalizeMcpTool(raw);
      if (tool) tools.push(tool);
      else invalid.push(invalidToolLabel(raw, index));
    });
    onProgress?.(`Normalized ${tools.length} of ${rawTools.length} raw MCP tools`);
    return { raw: rawTools.length, tools, invalid };
  } catch (error) {
    onProgress?.("MCP tools discovery encountered an error");
    if (parentSignal?.aborted) throw parentSignal.reason;
    throw error;
  }
}

function mcpErrorMessage(value: unknown, depth = 0): string | undefined {
  if (depth > MAX_ERROR_WALK_DEPTH) return undefined;
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    for (const entry of value) {
      const message = mcpErrorMessage(entry, depth + 1);
      if (message !== undefined) return message;
    }
    return undefined;
  }
  const record = asRecord(value);
  if (!record) return undefined;
  return (
    stringValue(record.message) ??
    (record.error !== value ? mcpErrorMessage(record.error, depth + 1) : undefined) ??
    stringValue(record.text)
  );
}

function serializeResult(value: unknown): string {
  const serialized = JSON.stringify(value, null, 2) ?? "null";
  return truncateUtf8(serialized, MAX_RESULT_BYTES, TRUNCATION_MARKER);
}

function mcpCallError(serverId: string, toolName: string, detail: string): Error {
  return new Error(
    truncateUtf8(`Error calling ${toolName} on ${serverId}: ${detail}`, MAX_RESULT_BYTES, TRUNCATION_MARKER),
  );
}

export async function executeMcpTool(
  baseUrl: string,
  apiKey: string,
  serverId: string,
  toolName: string,
  args: Record<string, unknown>,
  headers?: Record<string, string>,
  parentSignal?: AbortSignal,
): Promise<string> {
  parentSignal?.throwIfAborted();
  const signal = parentSignal
    ? AbortSignal.any([parentSignal, AbortSignal.timeout(CALL_TIMEOUT_MS)])
    : AbortSignal.timeout(CALL_TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${normalizeBaseUrl(baseUrl)}/mcp-rest/tools/call`, {
      method: "POST",
      headers: {
        ...headers,
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ server_id: serverId, name: toolName, arguments: args }),
      signal,
    });
    if (!response.ok) throw mcpCallError(serverId, toolName, `HTTP ${response.status}`);
    const body = parseJson(await readBoundedText(response, MAX_CALL_BODY_BYTES, "MCP tool call"), "MCP tool call");
    const bodyRecord = asRecord(body);
    if (bodyRecord?.error != null) {
      throw mcpCallError(serverId, toolName, mcpErrorMessage(bodyRecord.error) ?? "MCP error");
    }
    const result = bodyRecord && "result" in bodyRecord ? bodyRecord.result : body;
    const resultRecord = asRecord(result);
    if (resultRecord?.isError === true || bodyRecord?.isError === true) {
      const message =
        mcpErrorMessage(resultRecord?.content ?? bodyRecord?.content ?? result) ?? serializeResult(result);
      throw mcpCallError(serverId, toolName, message);
    }
    return serializeResult(result);
  } catch (error) {
    if (parentSignal?.aborted) throw parentSignal.reason;
    throw error;
  }
}

function sanitizeName(name: string): string {
  const sanitized = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return sanitized || "tool";
}

function toolIdentity(tool: LiteLLMMcpTool): string {
  return `${tool.server_id ?? tool.server_name}\0${tool.server_name}\0${tool.name}`;
}

/**
 * Derives a Pi tool name from the tool's identity alone.
 *
 * The hash is unconditional so the name is a pure function of `server_id`/`server_name`/`name`.
 * A conditional hash would make a survivor's name depend on which *other* tools happened to be in
 * the same catalog, so adding or removing a sibling would rename it — and because Pi cannot
 * unregister, the old name would linger and the model would see one tool twice.
 */
function buildPiToolName(tool: LiteLLMMcpTool): string {
  const hash = createHash("sha256").update(toolIdentity(tool)).digest("hex").slice(0, TOOL_NAME_HASH_LENGTH);
  const base = `mcp_${sanitizeName(tool.server_name)}_${sanitizeName(tool.name)}`;
  return `${base.slice(0, MAX_TOOL_NAME_LENGTH - hash.length - 1)}_${hash}`;
}

function schemaDepth(value: unknown, depth = 0): number {
  if (!value || typeof value !== "object") return depth;
  if (depth > MAX_SCHEMA_DEPTH) return depth;
  const values = Array.isArray(value) ? value : Object.values(value as Record<string, unknown>);
  return values.reduce((maximum, child) => Math.max(maximum, schemaDepth(child, depth + 1)), depth);
}

const DIRECT_SCHEMA_KEYWORDS = [
  "additionalProperties",
  "contains",
  "contentSchema",
  "else",
  "if",
  "not",
  "propertyNames",
  "then",
  "unevaluatedItems",
  "unevaluatedProperties",
] as const;
const SCHEMA_ARRAY_KEYWORDS = ["allOf", "anyOf", "oneOf", "prefixItems"] as const;
const SCHEMA_MAP_KEYWORDS = ["$defs", "definitions", "dependentSchemas", "patternProperties", "properties"] as const;

// TypeBox compiles a schema's `pattern` and the keys of `patternProperties` into a backtracking
// RegExp and executes them with no time limit, so a proxy-supplied expression is an unbounded CPU
// sink. A `$ref` resolves an arbitrary JSON pointer, which means a regex can be parked under any key
// at all — including data-only positions such as `default` or `examples`. Enumerating "positions
// where a subschema may appear" is therefore not a sound bound. We instead walk the entire supplied
// document, every key and every value, and refuse to vouch for anything we could not fully inspect.
const UNSAFE_REGEX_KEY = "pattern";
const UNSAFE_REGEX_MAP_KEY = "patternProperties";
const NONLOCAL_REF_KEYS = ["$dynamicRef", "$recursiveRef"] as const;
const MAX_SCAN_NODES = 20_000;
// Keywords whose immediate children are keyed by *name* rather than by keyword. Inside one of these
// a key like `patternProperties` is the name of a tool argument, not a constraint, so the keyword
// checks must not fire on it. This is the only context the scan needs; it does not restrict where
// the scan looks, only how a key is interpreted.
const NAME_KEYED_KEYWORDS = new Set([
  "properties",
  "patternProperties",
  "$defs",
  "definitions",
  "dependentSchemas",
  "dependencies",
]);

/** Why a supplied schema could not be proven safe to hand to the validator. */
export type SchemaHazard = "regex" | "nonlocal-ref" | "budget" | "cycle";

/**
 * Walks the whole untrusted schema graph looking for anything the validator could turn into a
 * regex, plus references we cannot resolve within the document we were given.
 *
 * Bounded three ways so a hostile document cannot exhaust us while we inspect it: a depth cap, a
 * node budget, and identity-based cycle detection. Hitting any bound returns a hazard rather than
 * `undefined`, because an unfinished scan proves nothing.
 *
 * A local `#/...` ref is safe precisely because this walk covers the whole document, so its target
 * has been inspected too. Any other ref form is a hazard.
 */
export function findSchemaHazard(root: Record<string, unknown>): SchemaHazard | undefined {
  const seen = new WeakSet<object>();
  let budget = MAX_SCAN_NODES;

  const walk = (value: unknown, depth: number, keysAreNames: boolean): SchemaHazard | undefined => {
    if (budget-- <= 0) return "budget";
    if (depth > MAX_SCHEMA_DEPTH) return "budget";
    if (value === null || typeof value !== "object") return undefined;
    if (seen.has(value)) return "cycle";
    seen.add(value);

    if (Array.isArray(value)) {
      for (const child of value) {
        const hazard = walk(child, depth + 1, false);
        if (hazard) return hazard;
      }
      return undefined;
    }

    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      if (!keysAreNames) {
        // A regex only matters where the validator would compile one: `pattern` holding a string,
        // or `patternProperties` holding a map whose keys are expressions.
        if (key === UNSAFE_REGEX_KEY && typeof child === "string") return "regex";
        if (key === UNSAFE_REGEX_MAP_KEY && asRecord(child) !== undefined) return "regex";
        // A local `#/...` pointer is safe because this walk covers the whole document, so its
        // target has been inspected too. Anything else points outside what we can see.
        if (key === "$ref" && typeof child === "string" && !child.startsWith("#")) return "nonlocal-ref";
        if ((NONLOCAL_REF_KEYS as readonly string[]).includes(key) && typeof child === "string") {
          return "nonlocal-ref";
        }
      }
      const hazard = walk(child, depth + 1, !keysAreNames && NAME_KEYED_KEYWORDS.has(key));
      if (hazard) return hazard;
    }
    return undefined;
  };

  return walk(root, 0, false);
}

function isValidSchema(value: unknown): boolean {
  if (typeof value === "boolean") return true;
  const schema = asRecord(value);
  return schema !== undefined && hasValidSchemaShape(schema);
}

function isValidSchemaMap(value: unknown): boolean {
  const schemas = asRecord(value);
  return schemas !== undefined && Object.values(schemas).every(isValidSchema);
}

function isValidSchemaArray(value: unknown): boolean {
  return Array.isArray(value) && value.every(isValidSchema);
}

function isValidRequired(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

// Shape validation is a separate concern from the hazard scan above: it exists to avoid handing the
// validator a structurally broken document, and it deliberately only looks where a subschema may
// legally appear so that data positions are not misread as constraints.
function hasValidSchemaShape(record: Record<string, unknown>): boolean {
  if ("required" in record && !isValidRequired(record.required)) return false;
  if (SCHEMA_MAP_KEYWORDS.some((keyword) => keyword in record && !isValidSchemaMap(record[keyword]))) return false;
  if (SCHEMA_ARRAY_KEYWORDS.some((keyword) => keyword in record && !isValidSchemaArray(record[keyword]))) return false;
  if (DIRECT_SCHEMA_KEYWORDS.some((keyword) => keyword in record && !isValidSchema(record[keyword]))) return false;
  const items = record.items;
  return !("items" in record) || isValidSchema(items) || isValidSchemaArray(items);
}

/** The extension-owned parameter shape. Nothing in it originates from the proxy. */
function trustedEnvelope(): TSchema {
  return Type.Object({
    args: Type.Record(Type.String(), Type.Unknown(), { description: "Tool arguments as key-value pairs" }),
  });
}

type BuiltParameters = {
  parameters: TSchema;
  syntheticArgsEnvelope: boolean;
  /** Set when the proxy supplied a usable schema that we replaced with the trusted envelope. */
  degraded?: McpDegradeReason;
};

function buildParameters(tool: LiteLLMMcpTool): BuiltParameters | McpDropReason {
  // An `inputSchema` that was present but not a JSON object is a malformed tool, not a schemaless
  // one, and must not be silently upgraded to the permissive envelope.
  if (tool.input_schema_malformed) return "invalid-schema";
  const inputSchema = tool.input_schema;
  // Absent or empty schema: the proxy asked for no parameters, which the trusted envelope models
  // exactly. This is the normal schemaless path, not a degradation.
  if (Object.keys(inputSchema).length === 0) {
    return { parameters: trustedEnvelope(), syntheticArgsEnvelope: true };
  }
  if (inputSchema.type !== "object") return "invalid-schema";
  let serialized: string;
  try {
    serialized = JSON.stringify(inputSchema);
  } catch {
    return "invalid-schema";
  }
  if (byteLength(serialized) > MAX_SCHEMA_BYTES || schemaDepth(inputSchema) > MAX_SCHEMA_DEPTH) {
    return "invalid-schema";
  }
  if (!hasValidSchemaShape(inputSchema)) return "invalid-schema";
  // Structurally fine, but we will only forward it if the whole graph is provably free of
  // proxy-supplied regexes and unresolvable references. Otherwise the tool still registers, with
  // the trusted envelope standing in for its schema.
  if (findSchemaHazard(inputSchema) !== undefined) {
    return { parameters: trustedEnvelope(), syntheticArgsEnvelope: true, degraded: "schema-envelope" };
  }
  return { parameters: Type.Unsafe(inputSchema), syntheticArgsEnvelope: false };
}

export function prepareTools(discovery: McpDiscovery): {
  prepared: PreparedTool[];
  report: McpPreparationReport;
} {
  const drops = new Map<McpDropReason, string[]>();
  const degrades = new Map<McpDegradeReason, string[]>();
  const recordDrop = (reason: McpDropReason, label: string): void => {
    drops.set(reason, [...(drops.get(reason) ?? []), label]);
  };

  for (const label of discovery.invalid) recordDrop("invalid-tool", label);

  const unique = new Map<string, LiteLLMMcpTool>();
  for (const tool of discovery.tools) {
    const identity = toolIdentity(tool);
    if (unique.has(identity)) {
      recordDrop("duplicate-identity", buildPiToolName(tool));
      continue;
    }
    unique.set(identity, tool);
  }

  const accepted: Array<Omit<PreparedTool, "name">> = [];
  for (const tool of unique.values()) {
    const built = buildParameters(tool);
    if (typeof built === "string") {
      recordDrop(built, buildPiToolName(tool));
      continue;
    }
    if (built.degraded) {
      degrades.set(built.degraded, [...(degrades.get(built.degraded) ?? []), buildPiToolName(tool)]);
    }
    accepted.push({ tool, parameters: built.parameters, syntheticArgsEnvelope: built.syntheticArgsEnvelope });
  }

  const candidates = accepted.slice(0, MAX_REGISTERED_TOOLS);
  const finalNames = new Set<string>();
  const prepared: PreparedTool[] = [];
  for (const candidate of candidates) {
    // Membership-independent: the name depends only on this tool's identity.
    const name = buildPiToolName(candidate.tool);
    if (finalNames.has(name)) {
      recordDrop("name-collision", name);
      continue;
    }
    finalNames.add(name);
    prepared.push({ ...candidate, name });
  }

  const envelopedNames = degrades.get("schema-envelope") ?? [];
  return {
    prepared,
    report: {
      discovered: discovery.raw,
      prepared: prepared.length,
      enveloped: envelopedNames.length,
      overflow: accepted.length - candidates.length,
      dropped: [...drops.entries()].map(([reason, tools]) => ({ reason, tools })),
      degraded: [...degrades.entries()].map(([reason, tools]) => ({ reason, tools })),
    },
  };
}

function emitPreparationDiagnostics(report: McpPreparationReport): void {
  const seen = new Set<string>();
  const emitClass = (reason: McpIncidentReason, message: string, tools: string[]): void => {
    seen.add(reason);
    // Identity covers the full membership, so a change beyond the printed sample still re-reports.
    emitSafetyDiagnostic(reason, message, `${tools.length}:${membershipIdentity(tools)}`);
  };
  for (const { reason, tools } of report.dropped) {
    emitClass(
      reason,
      `dropped ${plural(tools.length, "MCP tool")} with ${DROP_REASON_TEXT[reason]}: ${sampleList(tools)}.`,
      tools,
    );
  }
  for (const { reason, tools } of report.degraded) {
    emitClass(
      reason,
      `kept ${plural(tools.length, "MCP tool")} but replaced ${tools.length === 1 ? "its" : "their"} schema with a ` +
        `safe args envelope, because of ${DEGRADE_REASON_TEXT[reason]}: ${sampleList(tools)}.`,
      tools,
    );
  }
  if (report.overflow > 0) {
    seen.add("tool-cap");
    emitSafetyDiagnostic(
      "tool-cap",
      `ignoring ${plural(report.overflow, "MCP tool")} beyond the ${MAX_REGISTERED_TOOLS}-tool limit.`,
      `tool-cap:${report.overflow}`,
    );
  }
  // Clear classes that did not recur, so the same incident is reported again if it comes back
  // after a clean refresh instead of being suppressed as an unchanged identity.
  for (const reason of [...Object.keys(INCIDENT_REASON_TEXT), "tool-cap"]) {
    if (!seen.has(reason)) clearIncident(reason);
  }
}

export async function createMcpToolDefinitions(
  getAuth: (ctx?: ExtensionContext) => Promise<LiteLLMRuntimeAuth>,
  onProgress?: (message: string) => void,
  signal?: AbortSignal,
): Promise<{ definitions: ToolDefinition[]; report: McpPreparationReport }> {
  const discoveryAuth = await getAuth();
  const discovery = await discoverMcpTools(
    discoveryAuth.baseUrl,
    discoveryAuth.apiKey,
    discoveryAuth.headers,
    onProgress,
    signal,
  );
  const { prepared, report } = prepareTools(discovery);
  emitPreparationDiagnostics(report);
  const lostTotal = report.dropped.reduce((total, entry) => total + entry.tools.length, 0) + report.overflow;
  const lostDetail = [
    ...report.dropped.map((entry) => `${entry.reason}=${entry.tools.length}`),
    ...(report.overflow > 0 ? [`tool-cap=${report.overflow}`] : []),
  ].join(", ");
  // raw = prepared + lost, so an operator can always account for every entry the proxy returned.
  onProgress?.(
    `Prepared ${report.prepared} of ${report.discovered} raw MCP tools ` +
      `(${report.enveloped} kept with a safe args envelope); lost ${lostTotal}` +
      `${lostDetail ? ` (${lostDetail})` : ""}`,
  );

  const definitions = prepared.map(({ tool: mcpTool, name, parameters, syntheticArgsEnvelope }) => {
    const description = truncateUtf8(
      `${mcpTool.description} (via ${mcpTool.server_name} MCP server)`,
      MAX_DESCRIPTION_BYTES,
      DESCRIPTION_TRUNCATION_MARKER,
    );
    const promptSnippet = truncateUtf8(
      `${mcpTool.description} via ${mcpTool.server_name} MCP server`,
      MAX_DESCRIPTION_BYTES,
      DESCRIPTION_TRUNCATION_MARKER,
    );

    // `label` and the `details` fields below are proxy-supplied, so they are bounded like every other
    // untrusted string that reaches Pi's UI or the model.
    const label = truncateUtf8(`${mcpTool.server_name}: ${mcpTool.name}`, MAX_LABEL_BYTES, SHORT_TRUNCATION_MARKER);
    const boundedDetail = (value: string): string => truncateUtf8(value, MAX_DETAIL_BYTES, SHORT_TRUNCATION_MARKER);

    return defineTool({
      name,
      label,
      description,
      promptSnippet,
      executionMode: "parallel",
      parameters,
      async execute(_toolCallId, params: Static<typeof parameters>, toolSignal, _onUpdate, ctx) {
        const auth = await getAuth(ctx);
        const rawParams = params as Record<string, unknown>;
        const args = syntheticArgsEnvelope ? asRecord(rawParams.args) : rawParams;
        if (!args) throw new Error("Synthetic MCP tool arguments must contain an object-valued args property");
        const text = await executeMcpTool(
          auth.baseUrl,
          auth.apiKey,
          mcpTool.server_id ?? mcpTool.server_name,
          mcpTool.name,
          args,
          auth.headers,
          toolSignal,
        );
        return {
          content: [{ type: "text", text }],
          details: {
            server: boundedDetail(mcpTool.server_name),
            serverId: boundedDetail(mcpTool.server_id ?? mcpTool.server_name),
            tool: boundedDetail(mcpTool.name),
          },
        };
      },
    });
  });

  return { definitions, report };
}
