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
import { enrichCachedModel } from "./discover.js";
import { activeCredentialRoot, credentialRootResult, isPlaceholderHost, refreshRequired } from "./host-policy.js";
import { createLiteLLMProtocolApis, isLiteLLMApi, resolveModelBaseUrl } from "./protocols.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi } from "./types.js";

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  resolveCredentialRoot: (credential?: Credential, requestBaseUrl?: string) => string | undefined;
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

function modelHostError(model: Model<LiteLLMApi>, activeHost: string): Error | undefined {
  // A model whose api has no protocol implementation cannot be projected onto any
  // endpoint. Reject it individually so one malformed entry cannot fail the whole list.
  if (!isLiteLLMApi(model.api)) {
    return refreshRequired(`Cached model uses unsupported LiteLLM transport ${String(model.api)}`);
  }
  const stored = credentialRootResult(model.baseUrl, "Cached model");
  if ("error" in stored) return stored.error;
  if (isPlaceholderHost(stored.host)) {
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
  // `modelHostError` proves the api before this projection can index the protocol map.
  const error = modelHostError(model, active.host);
  if (error) throw error;
  return { ...model, provider, baseUrl: resolveModelBaseUrl(active.root, model.api) };
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  const reportedAvailabilityDiagnostics = new Set<string>();
  // Availability is re-evaluated constantly; report each distinct reason once.
  const reportUnavailable = (message: string): void => {
    if (reportedAvailabilityDiagnostics.has(message)) return;
    reportedAvailabilityDiagnostics.add(message);
    process.stderr.write(`LiteLLM (${options.id}): ${message}\n`);
  };
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
        const root = options.resolveCredentialRoot(credential);
        if (!root) return [];
        active = activeCredentialRoot(root);
      } catch (error) {
        reportUnavailable(error instanceof Error ? error.message : String(error));
        return [];
      }
      // Every rejection is per model and inside the guard, so no single malformed
      // entry can throw out of availability and empty every provider's model list.
      const available: Model<LiteLLMApi>[] = [];
      for (const model of models) {
        const error = modelHostError(model, active.host);
        if (error) {
          reportUnavailable(error.message);
          continue;
        }
        available.push({ ...model, baseUrl: resolveModelBaseUrl(active.root, model.api) });
      }
      return available;
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
          options.resolveCredentialRoot(undefined, requestOptions?.env?.LITELLM_BASE_URL),
        ),
        context,
        requestOptions,
      ),
    streamSimple: (model: Model<LiteLLMApi>, context: Context, requestOptions?: SimpleStreamOptions) =>
      provider.streamSimple(
        requestModel(
          options.id,
          model,
          options.resolveCredentialRoot(undefined, requestOptions?.env?.LITELLM_BASE_URL),
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
