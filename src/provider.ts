import {
  type ApiStreamOptions,
  type Context,
  type Credential,
  createProvider,
  type Model,
  type Provider,
  type ProviderAuth,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { enrichCachedModel, normalizeBaseUrl } from "./discover.js";
import { createLiteLLMProtocolApis, resolveModelBaseUrl } from "./protocols.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi } from "./types.js";

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  resolveCredentialRoot?: (credential?: Credential, requestBaseUrl?: string) => string | undefined;
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

export const DEFAULT_LITELLM_BASE_URL = "https://litellm.example.com";
const PLACEHOLDER_HOSTS = new Set([new URL(DEFAULT_LITELLM_BASE_URL).host]);

type ModelHostResult = { host: string } | { error: Error };

function refreshRequired(message: string): Error {
  return new Error(`${message}; a network refresh with a valid LiteLLM base URL is required`);
}

function rootHost(baseUrl: string, subject: string): string {
  try {
    const url = new URL(normalizeBaseUrl(baseUrl));
    if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("unsupported protocol");
    return url.host.toLowerCase();
  } catch {
    throw refreshRequired(`${subject} has an invalid LiteLLM model URL`);
  }
}

function modelHost(model: Model<LiteLLMApi>): ModelHostResult {
  try {
    return { host: rootHost(model.baseUrl, "Cached model") };
  } catch (error) {
    return { error: error instanceof Error ? error : refreshRequired("Cached model has an invalid LiteLLM model URL") };
  }
}

function activeCredentialRoot(root: string): { root: string; host: string } {
  if (root.trim() === "undefined") throw refreshRequired("Active credentials have an invalid LiteLLM model URL");
  const normalized = normalizeBaseUrl(root);
  const host = rootHost(normalized, "Active credentials");
  if (PLACEHOLDER_HOSTS.has(host)) {
    throw refreshRequired("Active credentials use a placeholder LiteLLM model host");
  }
  return { root: normalized, host };
}

function modelHostError(model: Model<LiteLLMApi>, activeHost: string): Error | undefined {
  const stored = modelHost(model);
  if ("error" in stored) return stored.error;
  if (PLACEHOLDER_HOSTS.has(stored.host)) {
    return refreshRequired("Cached model uses a placeholder LiteLLM model host");
  }
  if (stored.host !== activeHost) {
    return refreshRequired(
      `Cached model has stale LiteLLM model host ${stored.host}; active credentials use ${activeHost}`,
    );
  }
}

function requestModel(
  provider: string,
  model: Model<LiteLLMApi>,
  credentialRoot: string | undefined,
): Model<LiteLLMApi> {
  if (!credentialRoot) throw refreshRequired("Active credentials do not identify a LiteLLM model host");
  const active = activeCredentialRoot(credentialRoot);
  const error = modelHostError(model, active.host);
  if (error) throw error;
  return { ...model, provider, baseUrl: resolveModelBaseUrl(active.root, model.api) };
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  const reportedAvailabilityDiagnostics = new Set<string>();
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
      let active: { root: string; host: string };
      try {
        const root = options.resolveCredentialRoot?.(credential);
        if (!root) return [];
        active = activeCredentialRoot(root);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (!reportedAvailabilityDiagnostics.has(message)) {
          reportedAvailabilityDiagnostics.add(message);
          process.stderr.write(`LiteLLM (${options.id}): ${message}\n`);
        }
        return [];
      }
      return models
        .filter((model) => {
          const error = modelHostError(model, active.host);
          if (!error) return true;
          if (!reportedAvailabilityDiagnostics.has(error.message)) {
            reportedAvailabilityDiagnostics.add(error.message);
            process.stderr.write(`LiteLLM (${options.id}): ${error.message}\n`);
          }
          return false;
        })
        .map((model) => ({ ...model, baseUrl: resolveModelBaseUrl(active.root, model.api) }));
    },
    api: createLiteLLMProtocolApis(),
  });
  const refreshModels = provider.refreshModels;
  const guardedProvider: Provider<LiteLLMApi> = {
    ...provider,
    stream: <T extends LiteLLMApi>(model: Model<T>, context: Context, requestOptions?: ApiStreamOptions<T>) =>
      provider.stream(
        requestModel(
          options.id,
          model,
          options.resolveCredentialRoot?.(undefined, requestOptions?.env?.LITELLM_BASE_URL),
        ),
        context,
        requestOptions,
      ),
    streamSimple: (model: Model<LiteLLMApi>, context: Context, requestOptions?: SimpleStreamOptions) =>
      provider.streamSimple(
        requestModel(
          options.id,
          model,
          options.resolveCredentialRoot?.(undefined, requestOptions?.env?.LITELLM_BASE_URL),
        ),
        context,
        requestOptions,
      ),
  };
  if (!refreshModels) return guardedProvider;
  return {
    ...guardedProvider,
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
