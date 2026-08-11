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
    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow("exceeds 5 MiB");
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
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
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

  it("truncates returned result text to 64 KiB with a marker", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { result: "x".repeat(70 * 1024) }));

    const text = await executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {});

    expect(Buffer.byteLength(text)).toBeLessThanOrEqual(64 * 1024);
    expect(text).toContain("[truncated by pi-provider-litellm]");
  });
});

describe("createMcpToolDefinitions", () => {
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
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "search",
          server_name: "shared",
          server_id: "server-one",
          input_schema: { type: "object", properties: {} },
        },
        {
          name: "search",
          server_name: "shared",
          server_id: "server-two",
          input_schema: { type: "object", properties: {} },
        },
        {
          name: "search",
          server_name: "shared",
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
        { name: "array-root", server_name: "schema", input_schema: { type: "array", items: {} } },
        { name: "array-properties", server_name: "schema", input_schema: { type: "object", properties: [] } },
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

    expect(definitions.map((tool) => tool.name)).toEqual(["mcp_schema_valid"]);
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

  it("does not unwrap legitimate args properties", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, [
          {
            name: "takes-args",
            server_name: "server",
            input_schema: {
              type: "object",
              properties: { args: { type: "object", properties: { value: { type: "string" } } } },
            },
          },
        ]),
      )
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    await definition?.execute("call-1", { args: { value: "kept" } }, undefined, undefined, {} as never);

    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      body: JSON.stringify({
        server_id: "server",
        name: "takes-args",
        arguments: { args: { value: "kept" } },
      }),
    });
  });

  it("unwraps only a synthetic args envelope", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, [{ name: "unknown-schema", server_name: "server", input_schema: {} }]))
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
