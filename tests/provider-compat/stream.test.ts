import { type Context, getSupportedThinkingLevels, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { createCompatibilityHarness, RED_CIRCLE_PNG, sseChunk, successfulResponse } from "./helpers.js";

const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });

describe("native provider stream compatibility", () => {
  it("completes two text turns with usage", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("First")] };
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 2 } }),
    );

    const first = await models.streamSimple(model, context).result();
    expect(first.content).toEqual([{ type: "text", text: "Hello" }]);
    expect(first.usage.input).toBeGreaterThan(0);
    expect(first.usage.output).toBeGreaterThan(0);

    context.messages.push(first, user("Second"));
    respond(
      sseChunk({ choices: [{ delta: { content: "Again" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 1 } }),
    );
    const second = await models.streamSimple(model, context).result();
    expect(second.content).toEqual([{ type: "text", text: "Again" }]);
    expect(second.usage.input).toBeGreaterThan(0);
    expect(second.usage.output).toBeGreaterThan(0);
  });

  it("emits text start, delta, and end events", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1 } }),
    );

    const events: Array<{ type: string; delta?: string }> = [];
    for await (const event of models.streamSimple(model, { messages: [user("Hi")] })) {
      if (event.type === "text_delta") events.push({ type: event.type, delta: event.delta });
      else if (event.type === "text_start" || event.type === "text_end") events.push({ type: event.type });
    }

    expect(events).toEqual([{ type: "text_start" }, { type: "text_delta", delta: "Hello" }, { type: "text_end" }]);
  });

  it("assembles tool-call argument deltas", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({
        choices: [
          {
            delta: {
              tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "add", arguments: '{"a":' } }],
            },
            finish_reason: null,
          },
        ],
      }),
      sseChunk({
        choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "714}" } }] }, finish_reason: null }],
      }),
      sseChunk({
        choices: [{ delta: {}, finish_reason: "tool_calls" }],
        usage: { prompt_tokens: 2, completion_tokens: 3 },
      }),
    );

    const message = await models
      .streamSimple(model, {
        messages: [user("Add")],
        tools: [
          { name: "add", description: "Add", parameters: { type: "object", properties: { a: { type: "number" } } } },
        ],
      })
      .result();

    expect(message.content).toContainEqual({ type: "toolCall", id: "call_1", name: "add", arguments: { a: 714 } });
  });

  it.each([
    {
      name: "Kimi K2.6 binary thinking",
      backend: "moonshot/kimi-k2.6",
      params: ["thinking"],
      reasoning: "high" as const,
      expected: { thinking: { type: "enabled" } },
      absent: ["reasoning_effort", "include_reasoning", "reasoning_content", "merge_reasoning_content_in_choices"],
    },
    {
      name: "Kimi K3 effort",
      backend: "moonshot/kimi-k3",
      params: ["reasoning_effort"],
      reasoning: "max" as const,
      expected: { reasoning_effort: "max" },
      absent: ["thinking", "include_reasoning", "reasoning_content", "merge_reasoning_content_in_choices"],
    },
    {
      name: "DeepSeek V4 native controls",
      backend: "deepseek/deepseek-v4",
      params: ["thinking", "reasoning_effort"],
      reasoning: "max" as const,
      expected: { thinking: { type: "enabled" }, reasoning_effort: "max" },
      absent: ["include_reasoning", "reasoning_content", "merge_reasoning_content_in_choices"],
    },
    {
      name: "Azure Foundry DeepSeek effort only",
      backend: "azure_ai/deepseek-v4",
      params: ["reasoning_effort"],
      reasoning: "high" as const,
      expected: { reasoning_effort: "high" },
      absent: ["thinking", "include_reasoning", "reasoning_content", "merge_reasoning_content_in_choices"],
    },
  ])("serializes $name from discovered policy", async ({ backend, params, reasoning, expected, absent }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "evidence-route",
        litellm_params: { model: backend, allowed_openai_params: params },
        model_info: { id: "deployment", mode: "chat" },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning }).result();

    expect(requests[0]).toMatchObject(expected);
    for (const field of absent) expect(requests[0]).not.toHaveProperty(field);
  });

  it.each([
    { name: "Kimi K3", backend: "moonshot/kimi-k3" },
    { name: "DeepSeek V4", backend: "deepseek/deepseek-v4-pro" },
  ])("sends no speculative controls for evidence-absent $name", async ({ backend }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "evidence-absent-route",
        litellm_params: { model: backend },
        model_info: { id: "deployment", mode: "chat", supports_reasoning: true },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(model.reasoning).toBe(true);
    expect(getSupportedThinkingLevels(model)).toEqual([]);
    expect(requests).toHaveLength(1);
    expect(requests[0]).not.toHaveProperty("thinking");
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
  });

  it("serializes disabled Kimi K2.6 as binary thinking off", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "binary-route",
        litellm_params: { model: "moonshot/kimi-k2.6" },
        model_info: { id: "deployment", mode: "chat", supported_openai_params: ["thinking"] },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }).result();

    expect(requests[0]).toMatchObject({ thinking: { type: "disabled" } });
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
  });

  it("omits unsupported controls for Kimi K2.7 Code", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "code-route",
        litellm_params: { model: "moonshot/kimi-k2.7-code" },
        model_info: { id: "deployment", mode: "chat", supported_openai_params: ["thinking"] },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(requests[0]).not.toHaveProperty("thinking");
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
    expect(requests[0]).not.toHaveProperty("include_reasoning");
    expect(requests[0]).not.toHaveProperty("reasoning_content");
    expect(requests[0]).not.toHaveProperty("merge_reasoning_content_in_choices");
  });

  it.each([
    { source: "/model/info", rows: [{ model_name: "kimi-k3", model_info: { mode: "chat" } }] },
    { source: "/model/info fallback", rows: [{ model_name: "moonshotai/kimi-k2", model_info: { mode: "chat" } }] },
  ])("applies discovery-driven strict repair from $source for evidence-free Kimi routes", async ({ rows }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness(rows);
    respond(...successfulResponse("ok"));

    await models
      .streamSimple(model, {
        messages: [
          user("Call a tool"),
          {
            role: "assistant",
            content: [{ type: "toolCall", id: "call_1", name: "lookup", arguments: {} }],
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "toolUse",
            timestamp: 2,
          },
          {
            role: "toolResult",
            toolCallId: "call_1",
            toolName: "lookup",
            content: [{ type: "text", text: "result" }],
            isError: false,
            timestamp: 3,
          },
        ],
      })
      .result();

    expect(model).toMatchObject({ litellmPolicy: { normalizeStrictToolMessages: true } });
    expect(requests[0]?.messages).toContainEqual(expect.objectContaining({ role: "assistant", content: "" }));
  });

  it.each([
    { name: "Kimi K3", backend: "moonshot/kimi-k3", params: ["reasoning_effort"] },
    { name: "Kimi K2.7 Code", backend: "moonshot/kimi-k2.7-code", params: ["thinking"] },
    {
      name: "DeepSeek V4",
      backend: "deepseek/deepseek-v4",
      params: ["thinking", "reasoning_effort"],
    },
  ])("replays required reasoning content for $name", async ({ backend, params }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "replay-route",
        litellm_params: { model: backend },
        model_info: { mode: "chat", supported_openai_params: params },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models
      .streamSimple(model, {
        messages: [
          user("First"),
          {
            role: "assistant",
            content: [{ type: "text", text: "Answer" }],
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop",
            timestamp: 2,
          },
          user("Continue"),
        ],
      })
      .result();

    expect(requests[0]?.messages).toContainEqual(expect.objectContaining({ role: "assistant", reasoning_content: "" }));
  });

  it.each([
    {
      name: "Kimi K3",
      backend: "moonshot/kimi-k3",
      params: ["reasoning_effort"],
      expected: { reasoning_effort: "max" },
      absent: ["thinking"],
    },
    {
      name: "DeepSeek V4",
      backend: "deepseek/deepseek-v4",
      params: ["thinking", "reasoning_effort"],
      expected: { thinking: { type: "enabled" }, reasoning_effort: "max" },
      absent: [],
    },
  ])("rehydrates exact $name policy and wire behavior offline", async ({ backend, params, expected, absent }) => {
    const modelsStore = new InMemoryModelsStore();
    const rows = [
      {
        model_name: "cached-reasoning-route",
        litellm_params: { model: backend },
        model_info: { mode: "chat", supported_openai_params: params },
      },
    ];
    const online = await createCompatibilityHarness(rows, { modelsStore });
    online.respond(...successfulResponse("online"));
    await online.models.streamSimple(online.model, { messages: [user("Think")] }, { reasoning: "max" }).result();

    vi.restoreAllMocks();
    vi.resetModules();
    const offline = await createCompatibilityHarness(rows, { modelsStore, allowNetwork: false });
    offline.respond(...successfulResponse("offline"));
    await offline.models.streamSimple(offline.model, { messages: [user("Think")] }, { reasoning: "max" }).result();

    expect(offline.model).toEqual(online.model);
    expect(offline.requests[0]).toEqual(online.requests[0]);
    expect(offline.requests[0]).toMatchObject(expected);
    for (const field of absent) expect(offline.requests[0]).not.toHaveProperty(field);
  });

  it("handles thinking and a tool result across turns", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("Calculate")], tools: [] };
    respond(
      sseChunk({ choices: [{ delta: { reasoning_content: "714" }, finish_reason: null }] }),
      sseChunk({
        choices: [
          {
            delta: {
              tool_calls: [{ index: 0, id: "call_2", type: "function", function: { name: "lookup", arguments: "{}" } }],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 2 },
      }),
    );
    const first = await models.streamSimple(model, context, { reasoning: "high" }).result();
    expect(first.content).toContainEqual(expect.objectContaining({ type: "thinking", thinking: "714" }));

    context.messages.push(first, {
      role: "toolResult",
      toolCallId: "call_2",
      toolName: "lookup",
      content: [{ type: "text", text: "887" }],
      isError: false,
      timestamp: 2,
    });
    respond(
      sseChunk({ choices: [{ delta: { content: "887" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 4, completion_tokens: 1 } }),
    );
    const second = await models.streamSimple(model, context).result();

    expect(second.content).toContainEqual({ type: "text", text: "887" });
    expect(requests.at(-1)?.messages).toContainEqual(expect.objectContaining({ role: "tool", content: "887" }));
  });

  it("serializes image input", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({
        choices: [{ delta: { content: "red" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 2, completion_tokens: 1 },
      }),
    );

    await models
      .streamSimple(model, {
        messages: [
          { role: "user", content: [{ type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" }], timestamp: 1 },
        ],
      })
      .result();

    expect(requests[0]?.messages[0]?.content).toContainEqual({
      type: "image_url",
      image_url: { url: `data:image/png;base64,${RED_CIRCLE_PNG}` },
    });
  });
});
