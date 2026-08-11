import { type Credential, createProvider, type Model, type Provider, type ProviderAuth } from "@earendil-works/pi-ai";
import { enrichCachedModel, normalizeBaseUrl } from "./discover.js";
import { createLiteLLMProtocolApis, resolveModelBaseUrl } from "./protocols.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi } from "./types.js";

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  credentialBaseUrl?: (credential: Credential) => string | undefined;
  discover(credential: Credential, signal?: AbortSignal): Promise<DiscoveryResult & { baseUrl?: string }>;
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

const PLACEHOLDER_HOSTS = new Set(["litellm.example.com"]);

function modelHost(model: Model<LiteLLMApi>): string {
  return new URL(normalizeBaseUrl(model.baseUrl)).host.toLowerCase();
}

function projectModelsForCredential(
  provider: string,
  models: readonly Model<LiteLLMApi>[],
  credentialBaseUrl: string,
): Model<LiteLLMApi>[] {
  const activeRoot = normalizeBaseUrl(credentialBaseUrl);
  const activeHost = new URL(activeRoot).host.toLowerCase();
  if (PLACEHOLDER_HOSTS.has(activeHost)) {
    throw new Error(
      "Active credentials use a placeholder LiteLLM model host; a network refresh with a real host is required",
    );
  }
  for (const model of models) {
    const storedHost = modelHost(model);
    if (PLACEHOLDER_HOSTS.has(storedHost)) {
      throw new Error("Cached model uses a placeholder LiteLLM model host; a network refresh is required");
    }
    if (storedHost !== activeHost) {
      throw new Error(
        `Cached model has stale LiteLLM model host ${storedHost}; active credentials use ${activeHost}. A network refresh is required`,
      );
    }
  }
  return toNativeModels(provider, activeRoot, models);
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  const provider = createProvider<LiteLLMApi>({
    id: options.id,
    name: options.name,
    baseUrl: options.baseUrl,
    auth: options.auth,
    models: [],
    async fetchModels(context) {
      if (!context.credential) throw new Error("LiteLLM model discovery requires a credential");
      const result = await options.discover(context.credential, context.signal);
      return toNativeModels(options.id, result.baseUrl ?? options.baseUrl, result.models);
    },
    filterModels(models, credential) {
      const baseUrl = credential && options.credentialBaseUrl?.(credential);
      return baseUrl ? projectModelsForCredential(options.id, models, baseUrl) : models;
    },
    api: createLiteLLMProtocolApis(),
  });
  const refreshModels = provider.refreshModels;
  if (!refreshModels) return provider;
  return {
    ...provider,
    refreshModels: (context) =>
      refreshModels({
        ...context,
        store: {
          async read() {
            const entry = await context.store.read();
            return entry && { ...entry, models: entry.models.map(enrichCachedModel) };
          },
          write: (entry) => context.store.write(entry),
          delete: () => context.store.delete(),
        },
      }),
  };
}
