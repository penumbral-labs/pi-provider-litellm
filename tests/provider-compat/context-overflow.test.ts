import { isContextOverflow } from "@earendil-works/pi-ai";
import { expect, it } from "vitest";
import { createCompatibilityHarness, user } from "./helpers.js";

it.each([
  undefined,
  {
    model_name: "claude-sonnet",
    model_info: { mode: "chat", litellm_provider: "anthropic" },
    litellm_params: { model: "anthropic/claude-sonnet-4-6" },
  },
])("normalizes LiteLLM context overflow for Pi detection", async (route) => {
  const { models, model } = await createCompatibilityHarness(route);

  const message = await models.streamSimple(model, { messages: [user("Overflow the context")] }).result();

  expect(message.stopReason).toBe("error");
  expect(message.errorMessage).toContain("maximum context length");
  expect(isContextOverflow(message)).toBe(true);
});
