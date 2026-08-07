import {
  type Credential,
  createProvider,
  type Model,
  type Provider,
  type ProviderAuth,
} from "@earendil-works/pi-ai/compat";
import { createLiteLLMProtocolApis, resolveModelBaseUrl } from "./protocols.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi } from "./types.js";

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  credentialBaseUrl?: (credential: Credential) => string | undefined;
  discover(credential: Credential): Promise<DiscoveryResult & { baseUrl?: string }>;
};

export function toNativeModels(
  provider: string,
  baseUrl: string,
  models: readonly DiscoveredModel[],
): Model<LiteLLMApi>[] {
  return models.map((model) => ({
    ...model,
    provider,
    baseUrl: resolveModelBaseUrl(baseUrl, model.api),
  }));
}

function waitFor<T>(promise: Promise<T>, signal: AbortSignal): Promise<T | undefined> {
  if (signal.aborted) return Promise.resolve(undefined);
  return new Promise((resolve, reject) => {
    const onAbort = () => resolve(undefined);
    signal.addEventListener("abort", onAbort, { once: true });
    void promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  let discovery: Promise<DiscoveryResult & { baseUrl?: string }> | undefined;
  return createProvider<LiteLLMApi>({
    id: options.id,
    name: options.name,
    baseUrl: options.baseUrl,
    auth: options.auth,
    models: [],
    async fetchModels(context) {
      if (!context.credential) throw new Error("LiteLLM model discovery requires a credential");
      if (!discovery) {
        const current = options.discover(context.credential);
        const tracked = current.finally(() => {
          if (discovery === tracked) discovery = undefined;
        });
        discovery = tracked;
      }
      const result = await waitFor(discovery, context.signal);
      return result ? toNativeModels(options.id, result.baseUrl ?? options.baseUrl, result.models) : [];
    },
    filterModels(models, credential) {
      const baseUrl = credential && options.credentialBaseUrl?.(credential);
      return baseUrl ? toNativeModels(options.id, baseUrl, models) : models;
    },
    api: createLiteLLMProtocolApis(),
  });
}
