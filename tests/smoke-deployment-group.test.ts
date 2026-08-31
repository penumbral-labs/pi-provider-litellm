import { afterEach, describe, expect, it, vi } from "vitest";
import { captureGroupedDeployments, validateGroupedDeployments } from "../scripts/smoke-deployment-group.js";

function deployment(id: string, mode: string, modelName = "grouped-vidaimock") {
  return {
    model_name: modelName,
    model_info: {
      id,
      mode,
      supported_openai_params: ["temperature"],
    },
    litellm_params: {
      allowed_openai_params: ["reasoning_effort"],
    },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("validateGroupedDeployments", () => {
  it("returns the two grouped deployments when their ids are unique and their modes are exactly chat and responses", () => {
    const chat = deployment("deployment-chat", "chat");
    const responses = deployment("deployment-responses", "responses");

    expect(
      validateGroupedDeployments({
        data: [deployment("unrelated-responses", "responses", "vidaimock-responses"), chat, responses],
      }),
    ).toEqual([chat, responses]);
  });

  it("rejects duplicate deployment ids", () => {
    expect(() =>
      validateGroupedDeployments({
        data: [deployment("duplicate", "chat"), deployment("duplicate", "responses")],
      }),
    ).toThrow("grouped deployments must preserve unique model_info.id values");
  });

  it("does not let an unrelated responses deployment satisfy the grouped mode requirement", () => {
    expect(() =>
      validateGroupedDeployments({
        data: [
          deployment("grouped-chat-1", "chat"),
          deployment("grouped-chat-2", "chat"),
          deployment("unrelated-responses", "responses", "vidaimock-responses"),
        ],
      }),
    ).toThrow("grouped deployments must preserve exactly chat and responses modes");
  });

  it.each([
    [
      "model_info.id",
      {
        model_name: "grouped-vidaimock",
        model_info: { mode: "chat", supported_openai_params: [] },
        litellm_params: { allowed_openai_params: [] },
      },
    ],
    [
      "supported_openai_params",
      {
        model_name: "grouped-vidaimock",
        model_info: { id: "chat", mode: "chat" },
        litellm_params: { allowed_openai_params: [] },
      },
    ],
    [
      "allowed_openai_params",
      {
        model_name: "grouped-vidaimock",
        model_info: { id: "chat", mode: "chat", supported_openai_params: [] },
        litellm_params: {},
      },
    ],
  ])("rejects a grouped deployment missing %s", (field, invalidChat) => {
    expect(() => validateGroupedDeployments({ data: [invalidChat, deployment("responses", "responses")] })).toThrow(
      `grouped deployment is missing ${field}`,
    );
  });
});

describe("captureGroupedDeployments", () => {
  it("fetches /model/info with authentication and validates the live payload", async () => {
    const rows = [deployment("chat", "chat"), deployment("responses", "responses")];
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ data: rows }));

    await expect(captureGroupedDeployments("http://127.0.0.1:4000/", "sk-smoke", 1250)).resolves.toEqual(rows);
    expect(fetchSpy).toHaveBeenCalledWith("http://127.0.0.1:4000/model/info", {
      headers: { Authorization: "Bearer sk-smoke" },
      signal: expect.any(AbortSignal),
    });
  });
});
