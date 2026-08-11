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
import {
  createLiteLLMProtocolApis,
  isLiteLLMApi,
  resolveModelBaseUrl,
  SELECTABLE_LITELLM_API_NAMES,
} from "./protocols.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi } from "./types.js";

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  // Resolves the proxy root the given request/credential must target. `apiKey`
  // identifies the resolved credential on request paths, where no Credential is
  // available; implementations use it to scope any remembered root to the
  // credential it came from.
  resolveCredentialRoot?: (request: {
    credential?: Credential;
    requestBaseUrl?: string;
    apiKey?: string;
  }) => string | undefined;
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

// Shared with the extension entrypoint so both auth and request paths reject the
// same hosts.
export function isPlaceholderHost(host: string): boolean {
  return PLACEHOLDER_HOSTS.has(host.toLowerCase());
}

function refreshRequired(message: string): Error {
  return new Error(`${message}; a network refresh with a valid LiteLLM base URL is required`);
}

function parseHost(baseUrl: string): string | undefined {
  try {
    const url = new URL(normalizeBaseUrl(baseUrl));
    if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
    return url.host.toLowerCase();
  } catch {
    return undefined;
  }
}

function activeCredentialRoot(root: string): { root: string; host: string } {
  const host = parseHost(root);
  if (!host) throw refreshRequired("Active credentials have an invalid LiteLLM model URL");
  if (isPlaceholderHost(host)) throw refreshRequired("Active credentials use a placeholder LiteLLM model host");
  return { root: normalizeBaseUrl(root), host };
}

// Returns why a model cannot be requested with the active credentials, or
// undefined when it can be. Callers must reject before deriving a request URL:
// the protocol check here is what keeps resolveModelBaseUrl's registry lookup in
// bounds for models that came from user models.json config rather than from our
// own discovery.
function modelError(model: Model<LiteLLMApi>, activeHost: string): Error | undefined {
  if (!isLiteLLMApi(model.api)) {
    return new Error(
      `LiteLLM model ${model.id} declares unsupported protocol "${model.api}"; ` +
        `set "api" to one of ${SELECTABLE_LITELLM_API_NAMES.join(", ")} in models.json`,
    );
  }
  const storedHost = parseHost(model.baseUrl);
  if (!storedHost) return refreshRequired("Cached model has an invalid LiteLLM model URL");
  if (isPlaceholderHost(storedHost)) return refreshRequired("Cached model uses a placeholder LiteLLM model host");
  if (storedHost !== activeHost) {
    return refreshRequired(
      `Cached model has stale LiteLLM model host ${storedHost}; active credentials use ${activeHost}`,
    );
  }
  return undefined;
}

function requestModel(
  provider: string,
  model: Model<LiteLLMApi>,
  credentialRoot: string | undefined,
): Model<LiteLLMApi> {
  if (!credentialRoot) {
    throw refreshRequired("Active credentials do not identify a LiteLLM model host");
  }
  const active = activeCredentialRoot(credentialRoot);
  const error = modelError(model, active.host);
  if (error) throw error;
  return { ...model, provider, baseUrl: resolveModelBaseUrl(active.root, model.api) };
}

function requestRoot(
  options: LiteLLMProviderOptions,
  requestOptions: { apiKey?: string; env?: Record<string, string> } | undefined,
): string | undefined {
  return options.resolveCredentialRoot?.({
    requestBaseUrl: requestOptions?.env?.LITELLM_BASE_URL,
    apiKey: requestOptions?.apiKey,
  });
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  const reportedAvailabilityDiagnostics = new Set<string>();
  const reportOnce = (message: string): void => {
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
        const root = options.resolveCredentialRoot?.({ credential });
        if (!root) {
          // Staying silent here hid the single most common misconfiguration, but
          // an unconfigured install has no models and nothing to explain.
          if (models.length > 0) {
            reportOnce(
              `${models.length} model(s) hidden because no LiteLLM base URL is configured; ` +
                "set LITELLM_BASE_URL or run /login litellm",
            );
          }
          return [];
        }
        active = activeCredentialRoot(root);
      } catch (error) {
        reportOnce(error instanceof Error ? error.message : String(error));
        return [];
      }
      // flatMap, not filter+map: rejected models must never reach
      // resolveModelBaseUrl, whose registry lookup assumes a known protocol.
      return models.flatMap((model) => {
        const error = modelError(model, active.host);
        if (error) {
          reportOnce(error.message);
          return [];
        }
        return [{ ...model, baseUrl: resolveModelBaseUrl(active.root, model.api) }];
      });
    },
    api: createLiteLLMProtocolApis(),
  });
  const refreshModels = provider.refreshModels;
  const guardedProvider: Provider<LiteLLMApi> = {
    ...provider,
    stream: <T extends LiteLLMApi>(model: Model<T>, context: Context, requestOptions?: ApiStreamOptions<T>) =>
      provider.stream(requestModel(options.id, model, requestRoot(options, requestOptions)), context, requestOptions),
    streamSimple: (model: Model<LiteLLMApi>, context: Context, requestOptions?: SimpleStreamOptions) =>
      provider.streamSimple(
        requestModel(options.id, model, requestRoot(options, requestOptions)),
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
