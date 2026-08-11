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
const TRUNCATION_MARKER = "\n[truncated by pi-provider-litellm]";
const DESCRIPTION_TRUNCATION_MARKER = "… [truncated]";

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

const emittedDiagnostics = new Set<string>();

function emitSafetyDiagnostic(incident: string, message: string): void {
  if (emittedDiagnostics.has(incident)) return;
  emittedDiagnostics.add(incident);
  process.stderr.write(`LiteLLM MCP: ${message}\n`);
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
        await reader.cancel();
        emitSafetyDiagnostic(`${surface}-body-cap`, `${surface} response exceeded its ${limit}-byte limit.`);
        throw new Error(`${surface} response exceeds its ${limit}-byte limit`);
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
  if (depth > MAX_SCHEMA_DEPTH) return undefined;
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

function hasValidSchemaShape(value: unknown): boolean {
  if (Array.isArray(value)) return value.every(hasValidSchemaShape);
  const record = asRecord(value);
  if (!record) return true;
  if (
    "required" in record &&
    (!Array.isArray(record.required) || !record.required.every((item) => typeof item === "string"))
  ) {
    return false;
  }
  if ("properties" in record) {
    const properties = asRecord(record.properties);
    if (
      !properties ||
      !Object.values(properties).every((property) => asRecord(property) && hasValidSchemaShape(property))
    ) {
      return false;
    }
  }
  return Object.values(record).every(hasValidSchemaShape);
}

function buildParameters(
  inputSchema: Record<string, unknown>,
): { parameters: TSchema; syntheticArgsEnvelope: boolean } | undefined {
  if (Object.keys(inputSchema).length === 0) {
    return {
      parameters: Type.Object({
        args: Type.Record(Type.String(), Type.Unknown(), { description: "Tool arguments as key-value pairs" }),
      }),
      syntheticArgsEnvelope: true,
    };
  }
  if (inputSchema.type !== "object") return undefined;
  let serialized: string;
  try {
    serialized = JSON.stringify(inputSchema);
  } catch {
    return undefined;
  }
  if (
    byteLength(serialized) > MAX_SCHEMA_BYTES ||
    schemaDepth(inputSchema) > MAX_SCHEMA_DEPTH ||
    !hasValidSchemaShape(inputSchema)
  ) {
    return undefined;
  }
  return { parameters: Type.Unsafe(inputSchema), syntheticArgsEnvelope: false };
}

function prepareTools(tools: LiteLLMMcpTool[]): PreparedTool[] {
  const unique = new Map<string, LiteLLMMcpTool>();
  for (const tool of tools) {
    const identity = toolIdentity(tool);
    if (unique.has(identity)) {
      emitSafetyDiagnostic("duplicate-identity", "Ignored a duplicate MCP tool identity.");
      continue;
    }
    unique.set(identity, tool);
  }

  const candidates: Array<Omit<PreparedTool, "name">> = [];
  for (const tool of unique.values()) {
    const built = buildParameters(tool.input_schema);
    if (!built) {
      emitSafetyDiagnostic("invalid-schema", "Ignored an MCP tool with an invalid or oversized input schema.");
      continue;
    }
    candidates.push({ tool, ...built });
  }

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
      emitSafetyDiagnostic("name-collision", "Ignored an MCP tool whose generated Pi name collided.");
      continue;
    }
    finalNames.add(name);
    prepared.push({ ...candidate, name });
  }
  return prepared;
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
  const normalizedTools = prepareTools(tools);
  if (normalizedTools.length > MAX_REGISTERED_TOOLS) {
    emitSafetyDiagnostic(
      "tool-cap",
      `Ignoring ${normalizedTools.length - MAX_REGISTERED_TOOLS} MCP tools beyond the 512-tool limit.`,
    );
  }
  const prepared = normalizedTools.slice(0, MAX_REGISTERED_TOOLS);

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

    return defineTool({
      name,
      label: `${mcpTool.server_name}: ${mcpTool.name}`,
      description,
      promptSnippet,
      executionMode: "parallel",
      parameters,
      async execute(_toolCallId, params: Static<typeof parameters>, toolSignal, _onUpdate, ctx) {
        const auth = await getAuth(ctx);
        const rawParams = params as Record<string, unknown>;
        const args = syntheticArgsEnvelope ? (asRecord(rawParams.args) ?? {}) : rawParams;
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
          details: { server: mcpTool.server_name, serverId: mcpTool.server_id, tool: mcpTool.name },
        };
      },
    });
  });
}
