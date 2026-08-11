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

/** Why a discovered MCP tool was not registered. Each reason is reported as its own diagnostic class. */
export type McpDropReason = "duplicate-identity" | "invalid-schema" | "unsupported-pattern" | "name-collision";

const DROP_REASON_TEXT: Readonly<Record<McpDropReason, string>> = {
  "duplicate-identity": "duplicate identities",
  "invalid-schema": "an invalid or oversized input schema",
  "unsupported-pattern": "unsupported pattern constraints",
  "name-collision": "a colliding generated Pi name",
};

export interface McpPreparationReport {
  discovered: number;
  prepared: number;
  /** Tools discarded because the registered-tool cap was reached. */
  overflow: number;
  dropped: Array<{ reason: McpDropReason; tools: string[] }>;
}

// Incident-aware suppression: an incident class is re-reported whenever its message changes
// (different count or different tools), so a stable proxy stays quiet across refreshes while a
// changed one is always surfaced. Keys come from a fixed set of classes, so this cannot grow
// without bound.
const lastEmittedIncident = new Map<string, string>();

function emitSafetyDiagnostic(incident: string, message: string): void {
  if (lastEmittedIncident.get(incident) === message) return;
  lastEmittedIncident.set(incident, message);
  process.stderr.write(`LiteLLM MCP: ${message}\n`);
}

function plural(count: number, singular: string): string {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

/** Renders a bounded sample list. Names are already sanitized to `[a-z0-9_]`, so they are safe for stderr. */
function sampleList(names: readonly string[]): string {
  const shown = names.slice(0, MAX_DIAGNOSTIC_SAMPLES);
  const remainder = names.length - shown.length;
  return remainder > 0 ? `${shown.join(", ")} (+${remainder} more)` : shown.join(", ");
}

/**
 * Reports MCP tools that Pi itself refused to register. `cause` is Pi-authored (never proxy-supplied),
 * and tool names are sanitized identities, so neither the proxy's schema nor its response body can reach stderr.
 */
export function reportMcpRegistrationFailures(
  failures: ReadonlyArray<{ tool: string; cause: unknown }>,
  attempted: number,
): void {
  if (failures.length === 0) {
    lastEmittedIncident.delete("registration-failure");
    return;
  }
  const cause = failures[0]?.cause;
  const causeText = truncateUtf8(
    cause instanceof Error ? cause.message : String(cause),
    MAX_DIAGNOSTIC_CAUSE_BYTES,
    SHORT_TRUNCATION_MARKER,
  );
  emitSafetyDiagnostic(
    "registration-failure",
    `${failures.length} of ${plural(attempted, "MCP tool")} could not be registered: ` +
      `${sampleList(failures.map((failure) => failure.tool))} (first cause: ${causeText}).`,
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
  const inputSchema = asRecord(raw.inputSchema) ?? asRecord(raw.input_schema) ?? {};
  return {
    name,
    server_name: serverName,
    server_id: stringValue(raw.mcp_info?.server_id) ?? stringValue(raw.server_id) ?? serverName,
    description: stringValue(raw.description) ?? name,
    input_schema: inputSchema,
  };
}

export async function discoverMcpTools(
  baseUrl: string,
  apiKey: string,
  headers?: Record<string, string>,
  onProgress?: (message: string) => void,
  parentSignal?: AbortSignal,
): Promise<LiteLLMMcpTool[]> {
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
    const tools = rawTools.map(normalizeMcpTool).filter((tool): tool is LiteLLMMcpTool => tool !== undefined);
    onProgress?.(`Successfully discovered ${tools.length} MCP tools`);
    return tools;
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

function buildPiToolName(tool: LiteLLMMcpTool, forceHash = false): string {
  const base = `mcp_${sanitizeName(tool.server_name)}_${sanitizeName(tool.name)}`;
  if (!forceHash && base.length <= MAX_TOOL_NAME_LENGTH) return base;
  const hash = createHash("sha256").update(toolIdentity(tool)).digest("hex").slice(0, TOOL_NAME_HASH_LENGTH);
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

// `pattern` and the keys of `patternProperties` are compiled into a backtracking RegExp by TypeBox
// (schema/engine/pattern.mjs) and executed with no timeout, so a proxy-supplied regex is an unbounded
// CPU sink. Detection recurses over exactly the schema positions the validator walks.
const PATTERN_KEYWORDS = ["pattern", "patternProperties"] as const;

type SchemaVerdict = "ok" | "invalid-schema" | "unsupported-pattern";
type SchemaRejection = Extract<SchemaVerdict, "invalid-schema" | "unsupported-pattern">;

function worseVerdict(left: SchemaVerdict, right: SchemaVerdict): SchemaVerdict {
  if (left === "invalid-schema" || right === "invalid-schema") return "invalid-schema";
  if (left === "unsupported-pattern" || right === "unsupported-pattern") return "unsupported-pattern";
  return "ok";
}

function schemaVerdict(value: unknown, depth: number): SchemaVerdict {
  if (typeof value === "boolean") return "ok";
  const schema = asRecord(value);
  return schema === undefined ? "invalid-schema" : recordVerdict(schema, depth);
}

function schemaMapVerdict(value: unknown, depth: number): SchemaVerdict {
  const schemas = asRecord(value);
  if (schemas === undefined) return "invalid-schema";
  return Object.values(schemas).reduce<SchemaVerdict>(
    (verdict, child) => worseVerdict(verdict, schemaVerdict(child, depth)),
    "ok",
  );
}

function schemaArrayVerdict(value: unknown, depth: number): SchemaVerdict {
  if (!Array.isArray(value)) return "invalid-schema";
  return value.reduce<SchemaVerdict>((verdict, child) => worseVerdict(verdict, schemaVerdict(child, depth)), "ok");
}

function isValidRequired(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function recordVerdict(record: Record<string, unknown>, depth: number): SchemaVerdict {
  // Bounded here rather than relying on the caller's size/depth gates being evaluated first.
  if (depth > MAX_SCHEMA_DEPTH) return "invalid-schema";
  if ("required" in record && !isValidRequired(record.required)) return "invalid-schema";
  if (PATTERN_KEYWORDS.some((keyword) => keyword in record)) return "unsupported-pattern";
  const next = depth + 1;
  let verdict: SchemaVerdict = "ok";
  for (const keyword of SCHEMA_MAP_KEYWORDS) {
    if (keyword in record) verdict = worseVerdict(verdict, schemaMapVerdict(record[keyword], next));
  }
  for (const keyword of SCHEMA_ARRAY_KEYWORDS) {
    if (keyword in record) verdict = worseVerdict(verdict, schemaArrayVerdict(record[keyword], next));
  }
  for (const keyword of DIRECT_SCHEMA_KEYWORDS) {
    if (keyword in record) verdict = worseVerdict(verdict, schemaVerdict(record[keyword], next));
  }
  if ("items" in record) {
    const items = record.items;
    verdict = worseVerdict(
      verdict,
      Array.isArray(items) ? schemaArrayVerdict(items, next) : schemaVerdict(items, next),
    );
  }
  return verdict;
}

function buildParameters(
  inputSchema: Record<string, unknown>,
): { parameters: TSchema; syntheticArgsEnvelope: boolean } | SchemaRejection {
  if (Object.keys(inputSchema).length === 0) {
    return {
      parameters: Type.Object({
        args: Type.Record(Type.String(), Type.Unknown(), { description: "Tool arguments as key-value pairs" }),
      }),
      syntheticArgsEnvelope: true,
    };
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
  const verdict = recordVerdict(inputSchema, 0);
  if (verdict !== "ok") return verdict;
  return { parameters: Type.Unsafe(inputSchema), syntheticArgsEnvelope: false };
}

export function prepareTools(tools: LiteLLMMcpTool[]): {
  prepared: PreparedTool[];
  report: McpPreparationReport;
} {
  const drops = new Map<McpDropReason, string[]>();
  const recordDrop = (reason: McpDropReason, tool: LiteLLMMcpTool): void => {
    const names = drops.get(reason) ?? [];
    names.push(buildPiToolName(tool));
    drops.set(reason, names);
  };

  const unique = new Map<string, LiteLLMMcpTool>();
  for (const tool of tools) {
    const identity = toolIdentity(tool);
    if (unique.has(identity)) {
      recordDrop("duplicate-identity", tool);
      continue;
    }
    unique.set(identity, tool);
  }

  const accepted: Array<Omit<PreparedTool, "name">> = [];
  for (const tool of unique.values()) {
    const built = buildParameters(tool.input_schema);
    if (typeof built === "string") {
      recordDrop(built, tool);
      continue;
    }
    accepted.push({ tool, ...built });
  }

  // The cap is applied before naming so a tool that never registers cannot force a hash suffix onto a survivor.
  const candidates = accepted.slice(0, MAX_REGISTERED_TOOLS);
  const nameCounts = new Map<string, number>();
  for (const candidate of candidates) {
    const name = buildPiToolName(candidate.tool);
    nameCounts.set(name, (nameCounts.get(name) ?? 0) + 1);
  }
  const finalNames = new Set<string>();
  const prepared: PreparedTool[] = [];
  for (const candidate of candidates) {
    const ordinaryName = buildPiToolName(candidate.tool);
    const name = buildPiToolName(candidate.tool, (nameCounts.get(ordinaryName) ?? 0) > 1);
    if (finalNames.has(name)) {
      recordDrop("name-collision", candidate.tool);
      continue;
    }
    finalNames.add(name);
    prepared.push({ ...candidate, name });
  }

  return {
    prepared,
    report: {
      discovered: tools.length,
      prepared: prepared.length,
      overflow: accepted.length - candidates.length,
      dropped: [...drops.entries()].map(([reason, tools_]) => ({ reason, tools: tools_ })),
    },
  };
}

function emitPreparationDiagnostics(report: McpPreparationReport): void {
  const seen = new Set<string>();
  for (const { reason, tools } of report.dropped) {
    seen.add(reason);
    emitSafetyDiagnostic(
      reason,
      `dropped ${plural(tools.length, "MCP tool")} with ${DROP_REASON_TEXT[reason]}: ${sampleList(tools)}.`,
    );
  }
  if (report.overflow > 0) {
    seen.add("tool-cap");
    emitSafetyDiagnostic(
      "tool-cap",
      `ignoring ${plural(report.overflow, "MCP tool")} beyond the ${MAX_REGISTERED_TOOLS}-tool limit.`,
    );
  }
  // Clear classes that did not recur, so the same incident is reported again if it comes back
  // after a clean refresh instead of being suppressed as an unchanged message.
  for (const reason of [...Object.keys(DROP_REASON_TEXT), "tool-cap"]) {
    if (!seen.has(reason)) lastEmittedIncident.delete(reason);
  }
}

export async function createMcpToolDefinitions(
  getAuth: (ctx?: ExtensionContext) => Promise<LiteLLMRuntimeAuth>,
  onProgress?: (message: string) => void,
  signal?: AbortSignal,
): Promise<ToolDefinition[]> {
  const discoveryAuth = await getAuth();
  const tools = await discoverMcpTools(
    discoveryAuth.baseUrl,
    discoveryAuth.apiKey,
    discoveryAuth.headers,
    onProgress,
    signal,
  );
  const { prepared, report } = prepareTools(tools);
  emitPreparationDiagnostics(report);
  const droppedTotal = report.dropped.reduce((total, entry) => total + entry.tools.length, 0) + report.overflow;
  const droppedDetail = [
    ...report.dropped.map((entry) => `${entry.reason}=${entry.tools.length}`),
    ...(report.overflow > 0 ? [`tool-cap=${report.overflow}`] : []),
  ].join(", ");
  onProgress?.(
    `Prepared ${report.prepared} of ${report.discovered} discovered MCP tools; dropped ${droppedTotal}` +
      `${droppedDetail ? ` (${droppedDetail})` : ""}`,
  );

  return prepared.map(({ tool: mcpTool, name, parameters, syntheticArgsEnvelope }) => {
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
    const detail = (value: string): string => truncateUtf8(value, MAX_DETAIL_BYTES, SHORT_TRUNCATION_MARKER);

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
            server: detail(mcpTool.server_name),
            serverId: mcpTool.server_id === undefined ? undefined : detail(mcpTool.server_id),
            tool: detail(mcpTool.name),
          },
        };
      },
    });
  });
}
