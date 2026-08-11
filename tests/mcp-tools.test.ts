import type { Static, TSchema } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMcpToolDefinitions, discoverMcpTools, executeMcpTool } from "../src/mcp-tools.js";
import type { LiteLLMMcpTool } from "../src/types.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("discoverMcpTools", () => {
  it("returns tools from LiteLLM MCP REST discovery", async () => {
    const inputSchema = {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          {
            name: "web-search",
            description: "Search the web",
            inputSchema,
            mcp_info: { server_name: "Brave API", server_id: "brave-api" },
          },
        ],
      }),
    );

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).resolves.toEqual([
      {
        name: "web-search",
        server_name: "Brave API",
        server_id: "brave-api",
        description: "Search the web",
        input_schema: inputSchema,
      },
    ]);

    expect(fetchMock).toHaveBeenCalledWith(
      "https://litellm.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer sk-test" }),
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("keeps compatibility with older array-shaped discovery responses", async () => {
    const tools: LiteLLMMcpTool[] = [
      {
        name: "web-search",
        server_name: "Brave API",
        server_id: "Brave API",
        description: "Search the web",
        input_schema: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"],
        },
      },
    ];
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, tools));

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).resolves.toEqual(tools);

    expect(fetchMock).toHaveBeenCalledWith(
      "https://litellm.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer sk-test" }),
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("rejects malformed and oversized discovery bodies with actionable errors", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("<html>nope</html>", { status: 200 }))
      .mockResolvedValueOnce(new Response("x".repeat(5 * 1024 * 1024 + 1), { status: 200 }));

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow("invalid JSON");
    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow(
      "exceeds its 5242880-byte limit",
    );
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("returns the full bounded discovery list before registration filtering", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: Array.from({ length: 513 }, (_, index) => ({
          name: `tool-${index}`,
          server_name: "server",
          input_schema: { type: "object", properties: {} },
        })),
      }),
    );

    const tools = await discoverMcpTools("https://litellm.example.com", "sk-test");

    expect(tools).toHaveLength(513);
    expect(tools.at(-1)?.name).toBe("tool-512");
  });

  it("propagates the original discovery cancellation reason", async () => {
    const controller = new AbortController();
    const reason = new Error("refresh cancelled");
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
        }),
    );

    const discovery = discoverMcpTools(
      "https://litellm.example.com",
      "sk-test",
      undefined,
      undefined,
      controller.signal,
    );
    controller.abort(reason);

    await expect(discovery).rejects.toBe(reason);
  });

  it("rejects when discovery fails", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow("offline");
  });
});

describe("executeMcpTool", () => {
  it("calls LiteLLM MCP REST execution and formats the result", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { result: { content: [{ type: "text", text: "found" }] } }));

    await expect(
      executeMcpTool(
        "https://litellm.example.com",
        "sk-test",
        "brave",
        "search",
        { query: "pi" },
        { "X-Team": "agent" },
      ),
    ).resolves.toBe(JSON.stringify({ content: [{ type: "text", text: "found" }] }, null, 2));

    expect(fetchMock).toHaveBeenCalledWith(
      "https://litellm.example.com/mcp-rest/tools/call",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer sk-test", "X-Team": "agent" }),
        body: JSON.stringify({ server_id: "brave", name: "search", arguments: { query: "pi" } }),
      }),
    );
  });

  it.each([408, 425, 429, 500, 502, 503, 504])("attempts HTTP %i exactly once", async (status) => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(status, { error: "busy" }))
      .mockResolvedValueOnce(jsonResponse(200, { result: "must not run" }));

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", { query: "pi" }),
    ).rejects.toThrow(`HTTP ${status}`);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([
    new TypeError("fetch failed"),
    Object.assign(new Error("reset"), { code: "ECONNRESET" }),
    new DOMException("timed out", "TimeoutError"),
  ])("attempts transport failure $name exactly once", async (failure) => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(failure);

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", { query: "pi" }),
    ).rejects.toThrow();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("performs zero requests for a pre-aborted signal and propagates its reason", async () => {
    const controller = new AbortController();
    const reason = new Error("tool cancelled");
    controller.abort(reason);
    const fetchMock = vi.spyOn(globalThis, "fetch");

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {}, undefined, controller.signal),
    ).rejects.toBe(reason);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("propagates the original reason when cancelled during execution", async () => {
    const controller = new AbortController();
    const reason = new Error("tool cancelled");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new TypeError("fetch failed")), { once: true });
        }),
    );

    const execution = executeMcpTool(
      "https://litellm.example.com",
      "sk-test",
      "brave",
      "search",
      {},
      undefined,
      controller.signal,
    );
    controller.abort(reason);

    await expect(execution).rejects.toBe(reason);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("propagates the original reason when cancelled while reading the response body", async () => {
    const controller = new AbortController();
    const reason = new Error("tool cancelled after headers");
    const encoder = new TextEncoder();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      const body = new ReadableStream<Uint8Array>({
        start(streamController) {
          streamController.enqueue(encoder.encode('{"result":"partial'));
          init?.signal?.addEventListener("abort", () => streamController.error(new TypeError("body read failed")), {
            once: true,
          });
        },
      });
      return new Response(body, { status: 200 });
    });

    const execution = executeMcpTool(
      "https://litellm.example.com",
      "sk-test",
      "brave",
      "search",
      {},
      undefined,
      controller.signal,
    );
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(1));
    controller.abort(reason);

    await expect(execution).rejects.toBe(reason);
  });

  it("rejects response parse and MCP result errors", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("not json", { status: 200 }))
      .mockResolvedValueOnce(jsonResponse(200, { error: { message: "denied" } }))
      .mockResolvedValueOnce(jsonResponse(200, { result: { isError: true, content: [{ text: "failed" }] } }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "invalid JSON",
    );
    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "denied",
    );
    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "failed",
    );
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("accepts an explicit null MCP error as success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { error: null, result: "ok" }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).resolves.toBe(
      JSON.stringify("ok", null, 2),
    );
  });

  it("bounds tool-call bodies before parsing", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("x".repeat(5 * 1024 * 1024 + 1), { status: 200 }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "MCP tool call response exceeds its 5242880-byte limit",
    );
  });

  it("attempts a malformed successful body exactly once", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("not json", { status: 200 }))
      .mockResolvedValueOnce(jsonResponse(200, { result: "must not run" }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "invalid JSON",
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("truncates returned result text to 64 KiB with a marker without splitting multibyte text", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { result: "😀".repeat(20 * 1024) }));

    const text = await executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {});

    expect(Buffer.byteLength(text)).toBeLessThanOrEqual(64 * 1024);
    expect(text).toContain("[truncated by pi-provider-litellm]");
    expect(text).not.toContain("�");
  });

  it.each([
    { error: { message: "x".repeat(70 * 1024) } },
    { result: { isError: true, content: [{ text: "x".repeat(70 * 1024) }] } },
  ])("truncates MCP error text to 64 KiB", async (body) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, body));

    const error = await executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {}).catch(
      (failure: unknown) => failure,
    );

    expect(error).toBeInstanceOf(Error);
    expect(Buffer.byteLength((error as Error).message)).toBeLessThanOrEqual(64 * 1024);
    expect((error as Error).message).toContain("[truncated by pi-provider-litellm]");
  });

  it("bounds nested MCP error traversal", async () => {
    let error: unknown = "too deep";
    for (let depth = 0; depth < 100; depth += 1) error = [error];
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { error }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "MCP error",
    );
  });
});

describe("createMcpToolDefinitions", () => {
  it("reports each drop class separately with counts and sanitized names, and no credential or schema content", async () => {
    vi.resetModules();
    const {
      createMcpToolDefinitions: createDefinitions,
      discoverMcpTools: discoverTools,
      executeMcpTool: executeTool,
    } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const oversizedBody = "x".repeat(5 * 1024 * 1024 + 1);
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => new Response(oversizedBody, { status: 200 }));
    await expect(discoverTools("https://litellm.example.com", "sk-secret-must-not-leak")).rejects.toThrow();
    await expect(discoverTools("https://litellm.example.com", "sk-secret-must-not-leak")).rejects.toThrow();
    await expect(
      executeTool("https://litellm.example.com", "sk-secret-must-not-leak", "server", "tool", {}),
    ).rejects.toThrow();
    await expect(
      executeTool("https://litellm.example.com", "sk-secret-must-not-leak", "server", "tool", {}),
    ).rejects.toThrow();

    const privateSchemaText = "private-schema-value";
    const duplicate = {
      name: "duplicate",
      server_name: "server",
      server_id: "server-id",
      input_schema: { type: "object", properties: {} },
    };
    fetchMock.mockImplementation(async () =>
      jsonResponse(200, [
        duplicate,
        duplicate,
        {
          name: "bad-one",
          server_name: "server",
          input_schema: { type: "object", properties: { secret: privateSchemaText } },
        },
        {
          name: "bad-two",
          server_name: "server",
          input_schema: { type: "object", properties: { secret: "also-private" } },
        },
        ...Array.from({ length: 513 }, (_, index) => ({
          name: `valid-${index}`,
          server_name: "server",
          input_schema: { type: "object", properties: {} },
        })),
      ]),
    );

    await createDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-secret-must-not-leak",
    }));

    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics.filter((message) => message.includes("MCP discovery response exceeded"))).toHaveLength(1);
    expect(diagnostics.filter((message) => message.includes("MCP tool call response exceeded"))).toHaveLength(1);

    // Each class is reported on its own line, with a count and the sanitized generated names.
    const duplicateLine = diagnostics.find((message) => message.includes("duplicate identities"));
    expect(duplicateLine).toBe("LiteLLM MCP: dropped 1 MCP tool with duplicate identities: mcp_server_duplicate.\n");
    const schemaLine = diagnostics.find((message) => message.includes("invalid or oversized input schema"));
    expect(schemaLine).toBe(
      "LiteLLM MCP: dropped 2 MCP tools with an invalid or oversized input schema: mcp_server_bad_one, mcp_server_bad_two.\n",
    );
    // 513 valid + the surviving duplicate = 514 accepted, so exactly 2 exceed the cap.
    expect(diagnostics.filter((message) => message.includes("512-tool limit"))).toEqual([
      "LiteLLM MCP: ignoring 2 MCP tools beyond the 512-tool limit.\n",
    ]);

    expect(diagnostics.join("\n")).not.toContain("sk-secret-must-not-leak");
    expect(diagnostics.join("\n")).not.toContain(privateSchemaText);
    expect(diagnostics.join("\n")).not.toContain("also-private");
  });

  it("suppresses an unchanged drop report across refreshes but reports a changed one", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitions } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const invalid = (name: string) => ({ name, server_name: "server", input_schema: { type: "array" } });
    const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });
    const fetchMock = vi.spyOn(globalThis, "fetch");

    const schemaLines = () =>
      stderr.mock.calls
        .map(([message]) => String(message))
        .filter((message) => message.includes("invalid or oversized input schema"));

    fetchMock.mockImplementation(async () => jsonResponse(200, [invalid("bad-one")]));
    await createDefinitions(auth);
    expect(schemaLines()).toHaveLength(1);

    // Identical incident on the next refresh: stays quiet.
    await createDefinitions(auth);
    expect(schemaLines()).toHaveLength(1);

    // Count changed: reported again, because the operator's picture changed.
    fetchMock.mockImplementation(async () => jsonResponse(200, [invalid("bad-one"), invalid("bad-two")]));
    await createDefinitions(auth);
    expect(schemaLines()).toHaveLength(2);
    expect(schemaLines()[1]).toContain("dropped 2 MCP tools");

    // A clean refresh clears the class, so its recurrence is reported rather than suppressed as unchanged.
    fetchMock.mockImplementation(async () => jsonResponse(200, []));
    await createDefinitions(auth);
    expect(schemaLines()).toHaveLength(2);
    fetchMock.mockImplementation(async () => jsonResponse(200, [invalid("bad-one"), invalid("bad-two")]));
    await createDefinitions(auth);
    expect(schemaLines()).toHaveLength(3);
  });

  it("exposes prepared and dropped counts through verbose discovery progress", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "good", server_name: "server", input_schema: { type: "object", properties: {} } },
        { name: "bad", server_name: "server", input_schema: { type: "array" } },
      ]),
    );
    const progress: string[] = [];
    await createMcpToolDefinitions(
      async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" }),
      (message) => progress.push(message),
    );

    expect(progress).toContain("Prepared 1 of 2 discovered MCP tools; dropped 1 (invalid-schema=1)");
  });

  it("derives names from the post-cap set so a dropped sibling does not rename a survivor", async () => {
    const collidingTool = {
      name: "search",
      server_name: "shared",
      input_schema: { type: "object", properties: {} },
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { ...collidingTool, server_id: "server-one" },
        ...Array.from({ length: 511 }, (_, index) => ({
          name: `tool-${index}`,
          server_name: "server",
          input_schema: { type: "object", properties: {} },
        })),
        { ...collidingTool, server_id: "server-two" },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    // The second `shared/search` falls beyond the 512 cap and is never registered, so the survivor
    // keeps the plain readable name instead of being renamed by a tool that does not exist.
    expect(definitions).toHaveLength(512);
    expect(definitions[0]?.name).toBe("mcp_shared_search");
  });

  it("still disambiguates with a hash when both colliding tools survive the cap", async () => {
    const collidingTool = {
      name: "search",
      server_name: "shared",
      input_schema: { type: "object", properties: {} },
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { ...collidingTool, server_id: "server-one" },
        { ...collidingTool, server_id: "server-two" },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions).toHaveLength(2);
    for (const definition of definitions) expect(definition.name).toMatch(/^mcp_shared_search_[a-f0-9]{10}$/);
    expect(new Set(definitions.map((definition) => definition.name)).size).toBe(2);
  });

  it("caps registered tools after invalid entries are isolated", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "invalid", server_name: "server", input_schema: { type: "array" } },
        ...Array.from({ length: 513 }, (_, index) => ({
          name: `valid-${index}`,
          server_name: "server",
          input_schema: { type: "object", properties: {} },
        })),
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions).toHaveLength(512);
    expect(definitions.at(-1)?.name).toBe("mcp_server_valid_511");
  });

  it("creates sanitized Pi tool definitions with mapped parameter schemas", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "web-search",
          server_name: "Brave API",
          server_id: "brave-api",
          description: "Search the web",
          input_schema: {
            type: "object",
            properties: {
              query: { type: "string" },
              limit: { type: "integer" },
              safe: { type: "boolean" },
              tags: { type: "array", items: { type: "string" } },
            },
            required: ["query"],
          },
        },
      ] satisfies LiteLLMMcpTool[]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions.map((tool) => tool.name)).toEqual(["mcp_brave_api_web_search"]);
    expect(definitions[0]?.executionMode).toBe("parallel");
    expect(definitions[0]?.description).toBe("Search the web (via Brave API MCP server)");
    const parameters = definitions[0]?.parameters as { required?: string[]; properties?: Record<string, unknown> };
    expect(parameters.required).toEqual(["query"]);
    expect(Object.keys(parameters.properties ?? {})).toEqual(["query", "limit", "safe", "tags"]);
  });

  it("uses unique bounded names across server IDs and deduplicates exact identities", async () => {
    const longServerName = `shared-${"server".repeat(10)}`;
    const longToolName = `search-${"tool".repeat(12)}`;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: longToolName,
          server_name: longServerName,
          server_id: "server-one",
          input_schema: { type: "object", properties: {} },
        },
        {
          name: longToolName,
          server_name: longServerName,
          server_id: "server-two",
          input_schema: { type: "object", properties: {} },
        },
        {
          name: longToolName,
          server_name: longServerName,
          server_id: "server-one",
          input_schema: { type: "object", properties: {} },
        },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions).toHaveLength(2);
    expect(new Set(definitions.map((tool) => tool.name)).size).toBe(2);
    expect(definitions.every((tool) => tool.name.length <= 64)).toBe(true);
    expect(definitions.every((tool) => /^[a-z0-9_]+$/.test(tool.name))).toBe(true);
  });

  it("isolates invalid schemas and preserves valid siblings", async () => {
    const tooDeep: Record<string, unknown> = { type: "string" };
    for (let depth = 0; depth < 17; depth += 1) {
      Object.assign(tooDeep, { type: "object", properties: { nested: { ...tooDeep } } });
    }
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "valid", server_name: "schema", input_schema: { type: "object", properties: {} } },
        {
          name: "properties-argument",
          server_name: "schema",
          input_schema: {
            type: "object",
            properties: { properties: { type: "object", additionalProperties: { type: "string" } } },
          },
        },
        {
          name: "required-argument",
          server_name: "schema",
          input_schema: { type: "object", properties: { required: { type: "boolean" } } },
        },
        { name: "array-root", server_name: "schema", input_schema: { type: "array", items: {} } },
        { name: "array-properties", server_name: "schema", input_schema: { type: "object", properties: [] } },
        {
          name: "invalid-property",
          server_name: "schema",
          input_schema: { type: "object", properties: { query: "not-a-schema" } },
        },
        {
          name: "nested-invalid-property",
          server_name: "schema",
          input_schema: { type: "object", properties: { query: { type: "object", properties: [] } } },
        },
        {
          name: "invalid-required",
          server_name: "schema",
          input_schema: { type: "object", properties: {}, required: "query" },
        },
        {
          name: "non-string-required",
          server_name: "schema",
          input_schema: { type: "object", properties: {}, required: ["query", 1] },
        },
        { name: "deep", server_name: "schema", input_schema: tooDeep },
        {
          name: "large",
          server_name: "schema",
          input_schema: { type: "object", properties: { value: { description: "x".repeat(64 * 1024) } } },
        },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions.map((tool) => tool.name)).toEqual([
      "mcp_schema_valid",
      "mcp_schema_properties_argument",
      "mcp_schema_required_argument",
    ]);
  });

  it("accepts schemas at the depth and size boundaries", async () => {
    let acceptedDepth: Record<string, unknown> = { type: "string" };
    for (let depth = 0; depth < 7; depth += 1) {
      acceptedDepth = { type: "object", properties: { nested: acceptedDepth } };
    }
    const prefix = JSON.stringify({ type: "object", properties: { value: { description: "" } } }).length;
    const acceptedSize = {
      type: "object",
      properties: { value: { description: "x".repeat(64 * 1024 - prefix) } },
    };
    expect(Buffer.byteLength(JSON.stringify(acceptedSize))).toBe(64 * 1024);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "depth-boundary", server_name: "schema", input_schema: acceptedDepth },
        { name: "size-boundary", server_name: "schema", input_schema: acceptedSize },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions.map((tool) => tool.name)).toEqual(["mcp_schema_depth_boundary", "mcp_schema_size_boundary"]);
  });

  it("truncates descriptions and prompt snippets to 4 KiB", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "verbose",
          server_name: "server",
          description: "x".repeat(5 * 1024),
          input_schema: { type: "object", properties: {} },
        },
      ]),
    );

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(Buffer.byteLength(definition?.description ?? "")).toBeLessThanOrEqual(4 * 1024);
    expect(definition?.description).toContain("[truncated]");
    expect(Buffer.byteLength(definition?.promptSnippet ?? "")).toBeLessThanOrEqual(4 * 1024);
  });

  it("passes complex object schemas through to Pi tools", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "complex",
          server_name: "Schema",
          description: "Complex input",
          input_schema: {
            type: "object",
            properties: { nested: { type: "object", properties: { value: { type: "string" } } } },
            required: ["nested"],
          },
        },
      ] satisfies LiteLLMMcpTool[]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions[0]?.parameters).toMatchObject({
      properties: { nested: { type: "object", properties: { value: { type: "string" } } } },
      required: ["nested"],
    });
  });

  it.each([
    {
      label: "object",
      argsSchema: { type: "object", properties: { value: { type: "string" } } },
      value: { value: "kept" },
    },
    { label: "array", argsSchema: { type: "array", items: { type: "string" } }, value: ["kept"] },
  ])("does not unwrap legitimate $label-valued args properties", async ({ argsSchema, value }) => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, [
          {
            name: "takes-args",
            server_name: "server",
            input_schema: { type: "object", properties: { args: argsSchema } },
          },
        ]),
      )
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    await definition?.execute("call-1", { args: value }, undefined, undefined, {} as never);

    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      body: JSON.stringify({
        server_id: "server",
        name: "takes-args",
        arguments: { args: value },
      }),
    });
  });

  it("unwraps a synthetic args envelope for an absent schema", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, [{ name: "unknown-schema", server_name: "server" }]))
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    await definition?.execute("call-1", { args: { value: "unwrapped" } }, undefined, undefined, {} as never);

    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      body: JSON.stringify({ server_id: "server", name: "unknown-schema", arguments: { value: "unwrapped" } }),
    });
  });

  it.each([{ args: "bad" }, { args: ["bad"] }, {}])(
    "rejects malformed synthetic args without issuing a tool call",
    async (params) => {
      const fetchMock = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(jsonResponse(200, [{ name: "unknown-schema", server_name: "server" }]));
      const [definition] = await createMcpToolDefinitions(async () => ({
        baseUrl: "https://litellm.example.com",
        apiKey: "sk-test",
      }));

      await expect(definition?.execute("call-1", params, undefined, undefined, {} as never)).rejects.toThrow(
        "object-valued args property",
      );
      expect(fetchMock).toHaveBeenCalledTimes(1);
    },
  );

  it("propagates definition-level execution failures", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, [
          { name: "failure", server_name: "server", input_schema: { type: "object", properties: {} } },
        ]),
      )
      .mockResolvedValueOnce(jsonResponse(500, { error: "failed" }));
    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    await expect(definition?.execute("call-1", {}, undefined, undefined, {} as never)).rejects.toThrow("HTTP 500");
  });

  it("passes Pi cancellation through generated tool execution", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, [
          { name: "cancel", server_name: "server", input_schema: { type: "object", properties: {} } },
        ]),
      )
      .mockImplementationOnce(
        (_input, init) =>
          new Promise((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
          }),
      );
    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    const controller = new AbortController();
    const reason = new Error("cancelled by Pi");

    const execution = definition?.execute("call-1", {}, controller.signal, undefined, {} as never);
    controller.abort(reason);

    await expect(execution).rejects.toBe(reason);
  });

  it("uses a fresh token when a generated tool executes", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search",
              inputSchema: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        }),
      )
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));
    const getAuth = vi
      .fn()
      .mockResolvedValueOnce({ baseUrl: "https://old.example.com", apiKey: "discovery-token" })
      .mockResolvedValueOnce({
        baseUrl: "https://new.example.com",
        apiKey: "execution-token",
        headers: { "x-tenant": "new" },
      });

    const definitions = await createMcpToolDefinitions(getAuth);
    type Params = Static<TSchema>;
    const result = await definitions[0]?.execute(
      "call-1",
      { query: "pi" } as Params,
      undefined,
      undefined,
      {} as never,
    );

    expect(result?.content).toEqual([{ type: "text", text: JSON.stringify("ok", null, 2) }]);
    expect(getAuth).toHaveBeenCalledTimes(2);
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[0]).toBe("https://new.example.com/mcp-rest/tools/call");
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      headers: expect.objectContaining({ Authorization: "Bearer execution-token", "x-tenant": "new" }),
      body: JSON.stringify({ server_id: "brave-api", name: "search", arguments: { query: "pi" } }),
    });
  });
});

describe("body cap diagnostics under a failing cancel", () => {
  it("still emits the diagnostic and the cap error when reader.cancel() rejects", async () => {
    vi.resetModules();
    const { discoverMcpTools: discoverTools } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    // A body that breaches the cap and whose cancel() rejects, which is what an errored stream does.
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(5 * 1024 * 1024 + 1));
      },
      cancel() {
        return Promise.reject(new Error("upstream reset"));
      },
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(body, { status: 200 }));

    await expect(discoverTools("https://litellm.example.com", "sk-test")).rejects.toThrow(
      "MCP discovery response exceeds its 5242880-byte limit",
    );
    expect(stderr.mock.calls.map(([message]) => String(message))).toContain(
      "LiteLLM MCP: MCP discovery response exceeded its 5242880-byte limit.\n",
    );
  });
});

describe("proxy-supplied regex constraints", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });
  const CATASTROPHIC = "^(a+)+$";

  function collectKeys(value: unknown, found: string[] = []): string[] {
    if (Array.isArray(value)) {
      for (const child of value) collectKeys(child, found);
      return found;
    }
    if (value && typeof value === "object") {
      for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
        found.push(key);
        collectKeys(child, found);
      }
    }
    return found;
  }

  const patternPositions: Array<[string, unknown]> = [
    ["root property", { type: "object", properties: { s: { type: "string", pattern: CATASTROPHIC } } }],
    [
      "deeply nested property",
      {
        type: "object",
        properties: { outer: { type: "object", properties: { s: { type: "string", pattern: CATASTROPHIC } } } },
      },
    ],
    ["allOf branch", { type: "object", properties: { s: { allOf: [{ type: "string", pattern: CATASTROPHIC }] } } }],
    ["items schema", { type: "object", properties: { a: { type: "array", items: { pattern: CATASTROPHIC } } } }],
    ["$defs entry", { type: "object", $defs: { s: { type: "string", pattern: CATASTROPHIC } }, properties: {} }],
    ["patternProperties keys", { type: "object", patternProperties: { [CATASTROPHIC]: { type: "string" } } }],
    ["schema root", { type: "object", pattern: CATASTROPHIC, properties: {} }],
  ];

  it.each(patternPositions)(
    "drops a tool whose schema carries a regex constraint at the %s",
    async (_position, inputSchema) => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse(200, [
          { name: "evil", server_name: "srv", input_schema: inputSchema },
          { name: "benign", server_name: "srv", input_schema: { type: "object", properties: {} } },
        ]),
      );
      const definitions = await createMcpToolDefinitions(auth);

      // The valid sibling is preserved; only the regex-bearing tool is refused.
      expect(definitions.map((definition) => definition.name)).toEqual(["mcp_srv_benign"]);
      // The invariant that matters: the validator only ever sees `definition.parameters`, so if no
      // registered schema contains a regex keyword, no regex can be compiled or executed.
      for (const definition of definitions) {
        expect(collectKeys(definition.parameters)).not.toContain("pattern");
        expect(collectKeys(definition.parameters)).not.toContain("patternProperties");
      }
    },
  );

  it("reports regex-bearing drops under their own diagnostic class", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitions } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "lookup",
          server_name: "srv",
          input_schema: { type: "object", properties: { s: { type: "string", pattern: CATASTROPHIC } } },
        },
        { name: "broken", server_name: "srv", input_schema: { type: "array" } },
      ]),
    );

    await createDefinitions(auth);

    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toContain(
      "LiteLLM MCP: dropped 1 MCP tool with unsupported pattern constraints: mcp_srv_lookup.\n",
    );
    // Kept distinct from the generic invalid-schema class so the two are separately actionable.
    expect(diagnostics).toContain(
      "LiteLLM MCP: dropped 1 MCP tool with an invalid or oversized input schema: mcp_srv_broken.\n",
    );
    // The regex itself is proxy-supplied and never echoed.
    expect(diagnostics.join("\n")).not.toContain(CATASTROPHIC);
  });
});

describe("bounded untrusted display strings", () => {
  it("truncates oversized label and details values with a marker", async () => {
    const hugeServer = "s".repeat(4096);
    const hugeTool = "t".repeat(4096);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [{ name: hugeTool, server_name: hugeServer, server_id: "i".repeat(4096), input_schema: {} }]),
    );

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definition).toBeDefined();
    expect(Buffer.byteLength(definition?.label ?? "", "utf8")).toBeLessThanOrEqual(256);
    expect(definition?.label).toMatch(/…$/);

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { result: "ok" }));
    const result = await definition?.execute?.(
      "call-1",
      { args: {} } as never,
      undefined as never,
      () => {},
      {} as never,
    );
    const details = (result as { details: Record<string, string> }).details;
    for (const value of Object.values(details)) {
      expect(Buffer.byteLength(value, "utf8")).toBeLessThanOrEqual(256);
      expect(value).toMatch(/…$/);
    }
  });

  it("leaves label and details untouched when they are already short", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [{ name: "search", server_name: "brave", server_id: "brave-id", input_schema: {} }]),
    );
    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    expect(definition?.label).toBe("brave: search");
  });
});
