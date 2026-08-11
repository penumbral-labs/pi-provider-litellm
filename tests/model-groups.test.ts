import { describe, expect, it } from "vitest";
import { type CatalogResolution, reduceModelGroup } from "../src/model-groups.js";
import type { ModelInfoEntry } from "../src/types.js";

const catalog = new Map<string, CatalogResolution>([
  [
    "openai/gpt-4o",
    {
      provider: "openai",
      semanticFamily: "openai",
      reasoning: false,
      vision: true,
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 5, output: 15, cacheRead: 2.5, cacheWrite: 0 },
    },
  ],
  [
    "anthropic/claude-sonnet-4-6",
    {
      provider: "anthropic",
      semanticFamily: "claude",
      reasoning: true,
      vision: true,
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    },
  ],
  [
    "bedrock/anthropic.claude-sonnet-4-6",
    {
      provider: "amazon-bedrock",
      semanticFamily: "claude",
      reasoning: true,
      vision: true,
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    },
  ],
]);

const resolveCatalog = (entry: ModelInfoEntry) => {
  const backend = entry.litellm_params?.model ?? entry.model_info?.base_model;
  return backend ? catalog.get(backend) : undefined;
};

function row(overrides: Partial<ModelInfoEntry> = {}): ModelInfoEntry {
  const { model_info, litellm_params, ...entry } = overrides;
  return {
    model_name: "route",
    ...entry,
    model_info: {
      id: "deployment-a",
      mode: "chat",
      supports_reasoning: true,
      supports_vision: true,
      max_input_tokens: 200_000,
      max_output_tokens: 32_000,
      input_cost_per_token: 0.000003,
      output_cost_per_token: 0.000015,
      cache_read_input_token_cost: 0.0000003,
      cache_creation_input_token_cost: 0.00000375,
      ...model_info,
    },
    litellm_params: { model: "anthropic/claude-sonnet-4-6", ...litellm_params },
  };
}

function permutations<T>(values: readonly T[]): T[][] {
  if (values.length < 2) return [[...values]];
  return values.flatMap((value, index) =>
    permutations([...values.slice(0, index), ...values.slice(index + 1)]).map((rest) => [value, ...rest]),
  );
}

describe("reduceModelGroup", () => {
  it("is permutation invariant for heterogeneous deployment evidence", () => {
    const deployments = [
      row({ model_info: { id: "deployment-a", mode: "responses", max_input_tokens: 150_000 } }),
      row({
        model_info: { id: "deployment-b", mode: "chat", max_output_tokens: 16_000 },
        litellm_params: { model: "openai/gpt-4o" },
      }),
      row({
        model_info: { id: "deployment-c", mode: null, supported_openai_params: ["reasoning_effort"] },
        litellm_params: { model: "internal/unknown" },
      }),
      row({
        model_info: { id: undefined, mode: "chat", output_cost_per_token: 0.00002 },
        litellm_params: { model: "internal/unknown", allowed_openai_params: ["reasoning_effort"] },
      }),
    ];
    const expected = {
      id: "route",
      deploymentCount: 4,
      api: "openai-completions",
      reasoning: true,
      vision: true,
      contextWindow: 150_000,
      maxTokens: 16_000,
      cost: { input: 3, output: 20, cacheRead: 0.3, cacheWrite: 3.75 },
      hasCompleteCost: true,
      semanticFamily: "conflicting",
      catalogAuthorityAmbiguous: true,
      acceptedOpenAIParams: [],
      reasoningPolicy: { reasoning: false },
    };

    for (const order of permutations(deployments)) {
      expect(reduceModelGroup(order, resolveCatalog)).toEqual(expected);
    }
  });

  it("deduplicates exact rows and reduces conflicting duplicate ids conservatively", () => {
    const repeated = row();
    const conflicting = row({ model_info: { id: "deployment-a", mode: "chat", max_input_tokens: 8_000 } });
    const anonymous = row({ model_info: { id: undefined, mode: "chat" } });

    expect(reduceModelGroup([repeated, repeated], resolveCatalog)).toEqual(
      reduceModelGroup([repeated], resolveCatalog),
    );
    // Conflicting variants of one deployment id both stay in the reduction, so
    // the count reflects the rows actually reduced rather than distinct ids.
    // Counting them as one would re-admit the public route name as evidence.
    const expected = reduceModelGroup([repeated, conflicting], resolveCatalog);
    expect(expected).toMatchObject({ deploymentCount: 2, contextWindow: 8_000 });
    expect(reduceModelGroup([conflicting, repeated], resolveCatalog)).toEqual(expected);
    expect(reduceModelGroup([anonymous, anonymous], resolveCatalog)?.deploymentCount).toBe(2);
  });

  it("selects Responses only when every deployment explicitly reports it", () => {
    const responses = row({ model_info: { id: "responses", mode: "responses" } });
    const response = row({ model_info: { id: "response", mode: "response" } });
    const chat = row({ model_info: { id: "chat", mode: "chat" } });
    const unknown = row({ model_info: { id: "unknown", mode: null } });

    expect(reduceModelGroup([responses, response], resolveCatalog)?.api).toBe("openai-responses");
    expect(reduceModelGroup([responses, chat], resolveCatalog)?.api).toBe("openai-completions");
    expect(reduceModelGroup([responses, unknown], resolveCatalog)?.api).toBe("openai-completions");
  });

  it("lets unsupported transport evidence force Chat without affecting metadata", () => {
    const responses = row({ model_info: { id: "responses", mode: "responses" } });
    const unsupported = row({
      model_info: { id: "embed", mode: "embedding", max_input_tokens: 1, max_output_tokens: 1 },
      litellm_params: { model: "internal/embedding" },
    });

    expect(reduceModelGroup([responses, unsupported], resolveCatalog)).toMatchObject({
      api: "openai-completions",
      contextWindow: 200_000,
      maxTokens: 32_000,
    });
  });

  it("falls back to Chat without reducing metadata from unsupported rows", () => {
    const result = reduceModelGroup(
      [
        row(),
        row({
          model_info: {
            id: "embed",
            mode: "embedding",
            max_input_tokens: 8_000,
            max_output_tokens: 1,
            input_cost_per_token: undefined,
          },
          litellm_params: { model: "internal/embedding" },
        }),
      ],
      resolveCatalog,
    );
    expect(result).toMatchObject({
      api: "openai-completions",
      // Only the routable deployment is reduced, so an unsupported sibling no
      // longer inflates the count or suppresses the singleton catalog hint.
      deploymentCount: 1,
      contextWindow: 200_000,
      maxTokens: 32_000,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    });
  });

  it("drops a group when every deployment is non-chat", () => {
    expect(
      reduceModelGroup(
        [
          row({ model_info: { id: "embed-a", mode: "embedding" } }),
          row({ model_info: { id: "embed-b", mode: "embedding" } }),
        ],
        resolveCatalog,
      ),
    ).toBeUndefined();
  });

  it.each([
    [[true, true], true],
    [[true, false], false],
    [[true, undefined], false],
  ] as const)("reduces capability guarantees %j to %s", (values, expected) => {
    const deployments = values.map((value, index) =>
      row({
        model_info: { id: `deployment-${index}`, mode: "chat", supports_vision: value },
        ...(value === undefined ? { litellm_params: { model: "internal/unknown" } } : {}),
      }),
    );
    expect(reduceModelGroup(deployments, resolveCatalog)?.vision).toBe(expected);
  });

  it("resolves deployment limits before taking the safe group minimum", () => {
    const explicit = row({
      model_info: { id: "explicit", mode: "chat", max_input_tokens: 100_000, max_output_tokens: 8_000 },
    });
    const fromCatalog = row({
      model_info: { id: "catalog", mode: "chat", max_input_tokens: undefined, max_output_tokens: undefined },
    });
    const unknown = row({
      model_info: { id: "unknown", mode: "chat", max_input_tokens: undefined, max_output_tokens: undefined },
      litellm_params: { model: "internal/unknown" },
    });

    expect(reduceModelGroup([explicit, fromCatalog], resolveCatalog)).toMatchObject({
      contextWindow: 100_000,
      maxTokens: 8_000,
    });
    expect(reduceModelGroup([explicit, unknown], resolveCatalog)).toMatchObject({
      contextWindow: 100_000,
      maxTokens: 8_000,
    });
    expect(reduceModelGroup([unknown], resolveCatalog)).toMatchObject({
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
  });

  it("uses the maximum complete display price and marks incomplete price evidence unknown", () => {
    const cheaper = row({
      model_info: {
        id: "cheap",
        mode: "chat",
        input_cost_per_token: 0,
        output_cost_per_token: 0.00001,
        cache_read_input_token_cost: 0.0000002,
        cache_creation_input_token_cost: 0.000003,
      },
    });
    const pricier = row({
      model_info: {
        id: "pricey",
        mode: "chat",
        input_cost_per_token: 0.000004,
        output_cost_per_token: 0.00002,
        cache_read_input_token_cost: 0.0000004,
        cache_creation_input_token_cost: 0.000004,
      },
    });
    const complete = reduceModelGroup([cheaper, pricier], resolveCatalog);
    expect(complete).toMatchObject({
      hasCompleteCost: true,
      cost: { input: 4, output: 20, cacheWrite: 4 },
    });
    expect(complete?.cost.cacheRead).toBeCloseTo(0.4);

    const incomplete = row({
      model_info: {
        id: "incomplete",
        mode: "chat",
        input_cost_per_token: 0.000004,
        output_cost_per_token: undefined,
      },
      litellm_params: { model: "internal/unknown" },
    });
    expect(reduceModelGroup([cheaper, incomplete], resolveCatalog)).toMatchObject({
      hasCompleteCost: false,
      cost: { input: 4, output: 0, cacheRead: 0.3, cacheWrite: 3.75 },
    });
  });

  it("keeps semantic family evidence when no catalog identity resolves", () => {
    const result = reduceModelGroup(
      [
        row({
          model_name: "public-route",
          model_info: { id: "foundry", mode: "chat", litellm_provider: "azure_ai" },
          litellm_params: { model: "azure_ai/DeepSeek-V4", allowed_openai_params: ["reasoning_effort"] },
        }),
      ],
      () => ({ semanticFamily: "deepseek" }),
    );

    expect(result).toMatchObject({
      semanticFamily: "deepseek",
      acceptedOpenAIParams: ["reasoning_effort"],
    });
    expect(result).not.toHaveProperty("catalogProvider");
  });

  it("keeps semantic family separate from catalog provider identity", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { id: "bedrock", mode: "chat", litellm_provider: "bedrock" },
          litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-6" },
        }),
      ],
      resolveCatalog,
    );

    expect(result).toMatchObject({ catalogProvider: "amazon-bedrock", semanticFamily: "claude" });
  });

  it("uses catalog thinking maps for unambiguous identities without a known semantic family", () => {
    const thinkingLevelMap = { low: "low", high: "high" } as const;
    const result = reduceModelGroup([row()], () => ({
      provider: "xai",
      reasoning: true,
      thinkingLevelMap,
      vision: false,
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
    }));

    expect(result?.thinkingLevelMap).toEqual(thinkingLevelMap);
  });

  it("disables catalog authority for conflicting provider identities", () => {
    const result = reduceModelGroup(
      [
        row({ model_info: { id: "anthropic", mode: "chat" } }),
        row({
          model_info: { id: "openai", mode: "chat" },
          litellm_params: { model: "openai/gpt-4o" },
        }),
      ],
      resolveCatalog,
    );

    expect(result).not.toHaveProperty("catalogProvider");
    // Reported as contradictory rather than absent, so callers cannot treat the
    // route name as a remaining source of family evidence.
    expect(result?.semanticFamily).toBe("conflicting");
    expect(result?.thinkingLevelMap).toBeUndefined();
  });

  it("intersects accepted parameters across deployments", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { id: "a", mode: "chat", supported_openai_params: ["temperature", "reasoning_effort"] },
          litellm_params: { model: "internal/a" },
        }),
        row({
          model_info: { id: "b", mode: "chat", supported_openai_params: ["reasoning_effort", "thinking"] },
          litellm_params: { model: "internal/b", allowed_openai_params: ["reasoning_effort"] },
        }),
      ],
      resolveCatalog,
    );

    expect(result?.acceptedOpenAIParams).toEqual(["reasoning_effort"]);
  });

  it.each([
    {
      name: "Kimi K2.6 with binary thinking",
      semanticModel: "kimi-k2.5-k2.6" as const,
      params: ["thinking"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: "off",
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: null,
        },
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
      },
    },
    {
      // K2.7 Code cannot be switched off, so `off` stays denied while `high`
      // rides the accepted `thinking` param.
      name: "Kimi K2.7 Code with accepted thinking",
      semanticModel: "kimi-k2.7-code" as const,
      params: ["thinking"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: null,
        },
        compat: {
          supportsReasoningEffort: false,
          requiresReasoningContentOnAssistantMessages: true,
          thinkingFormat: "deepseek",
        },
      },
    },
    {
      name: "Kimi K2.7 Code without accepted controls",
      semanticModel: "kimi-k2.7-code" as const,
      params: undefined,
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: null,
          xhigh: null,
          max: null,
        },
        compat: { supportsReasoningEffort: false, requiresReasoningContentOnAssistantMessages: true },
      },
    },
    {
      name: "Kimi K2.6 without accepted controls",
      semanticModel: "kimi-k2.5-k2.6" as const,
      params: undefined,
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: null,
          xhigh: null,
          max: null,
        },
        compat: { supportsReasoningEffort: false },
      },
    },
    {
      name: "Kimi K3 with effort",
      semanticModel: "kimi-k3" as const,
      params: ["reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: "low",
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: {
          thinkingFormat: "openai",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
    {
      name: "DeepSeek V4 with native controls",
      semanticModel: "deepseek-v4" as const,
      params: ["thinking", "reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: "off",
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: {
          thinkingFormat: "deepseek",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
    {
      name: "DeepSeek V4 through an effort-only route",
      semanticModel: "deepseek-v4" as const,
      params: ["reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: {
          thinkingFormat: "openai",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
  ])("derives $name policy from semantic and accepted-control evidence", ({ semanticModel, params, expected }) => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supported_openai_params: params },
          litellm_params: { model: "internal/model" },
        }),
      ],
      () => ({ semanticFamily: semanticModel.startsWith("kimi") ? "kimi" : "deepseek", semanticModel }),
    );

    expect(result?.reasoningPolicy).toEqual(expected);
  });

  it.each([
    {
      name: "Kimi K3",
      semanticModel: "kimi-k3" as const,
      family: "kimi" as const,
    },
    {
      name: "DeepSeek V4",
      semanticModel: "deepseek-v4" as const,
      family: "deepseek" as const,
    },
  ])("preserves $name capability and replay without accepted-control evidence", ({ semanticModel, family }) => {
    const result = reduceModelGroup(
      [row({ model_info: { supports_reasoning: true }, litellm_params: { model: `internal/${semanticModel}` } })],
      () => ({ semanticFamily: family, semanticModel }),
    );

    expect(result?.reasoningPolicy).toEqual({
      reasoning: true,
      thinkingLevelMap: { off: null, minimal: null, low: null, medium: null, high: null, xhigh: null, max: null },
      compat: {
        requiresReasoningContentOnAssistantMessages: true,
        supportsReasoningEffort: false,
      },
    });
  });

  it("lets any explicit reasoning denial override accepted-control promotion", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { id: "denied", supports_reasoning: false, supported_openai_params: ["thinking"] },
          litellm_params: { model: "moonshot/kimi-k2.6" },
        }),
        row({
          model_info: { id: "accepted", supports_reasoning: true, supported_openai_params: ["thinking"] },
          litellm_params: { model: "moonshot/kimi-k2.6" },
        }),
      ],
      () => ({ semanticFamily: "kimi", semanticModel: "kimi-k2.5-k2.6", reasoning: true }),
    );

    expect(result?.reasoning).toBe(false);
    expect(result?.reasoningPolicy).toEqual({ reasoning: false, compat: { supportsReasoningEffort: false } });
  });

  it("lets explicit unanimous reasoning denial override the K2.7 Code contract", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: false, supported_openai_params: ["thinking"] },
          litellm_params: { model: "moonshot/kimi-k2.7-code" },
        }),
      ],
      () => ({ semanticFamily: "kimi", semanticModel: "kimi-k2.7-code", reasoning: true }),
    );

    expect(result?.reasoning).toBe(false);
    expect(result?.reasoningPolicy).toEqual({
      reasoning: false,
      compat: { supportsReasoningEffort: false, requiresReasoningContentOnAssistantMessages: true },
    });
  });

  it("fails closed for mixed semantic generations and accepted controls", () => {
    const deployments = [
      row({
        model_info: { id: "k2", supported_openai_params: ["thinking"] },
        litellm_params: { model: "moonshot/kimi-k2.6" },
      }),
      row({
        model_info: { id: "k3", supported_openai_params: ["reasoning_effort"] },
        litellm_params: { model: "moonshot/kimi-k3" },
      }),
    ];
    const result = reduceModelGroup(deployments, (entry) => ({
      semanticFamily: "kimi",
      semanticModel: entry.model_info?.id === "k2" ? "kimi-k2.5-k2.6" : "kimi-k3",
    }));

    expect(result?.acceptedOpenAIParams).toEqual([]);
    expect(result?.reasoningPolicy).toEqual({ reasoning: false });
  });
});
