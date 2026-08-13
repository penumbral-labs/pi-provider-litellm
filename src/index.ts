import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type {
  Api,
  ApiKeyCredential,
  AssistantMessage,
  AuthInteraction,
  Credential,
  OAuthCredential,
  OAuthCredentials,
  ProviderAuth,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, isGpt55Model, normalizeBaseUrl } from "./discover.js";
import { getGcloudToken, hasGcloudAdcCredentials, isGcloudTokenAuthEnabled } from "./gcloud-token.js";
import { getSessionIdFromFile } from "./litellm.js";
import { createMcpToolDefinitions } from "./mcp-tools.js";
import { createLiteLLMProvider, DEFAULT_LITELLM_BASE_URL, isPlaceholderHost } from "./provider.js";
import { createSkillsPromptSection, createSkillToolDefinitions, listSkills } from "./skills.js";
import type { DiscoveryOptions, LiteLLMModelPolicy, LiteLLMRuntimeAuth, ResolvedCredentials } from "./types.js";

const PROVIDER_NAME = "litellm";
const SETTINGS_KEY = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const GCLOUD_ADC_SOURCE = "gcloud ADC";
const ENV_API_KEY_HELPER = "LITELLM_API_KEY_HELPER";
const ENV_HEADERS = "LITELLM_HEADERS";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const ENV_VERBOSE_DISCOVERY = "LITELLM_VERBOSE_DISCOVERY";
const ENV_MODELS_DEV = "LITELLM_MODELS_DEV";
const DEFAULT_TIMEOUT_MS = 5000;
const LOGIN_TIMEOUT_MS = 10_000;
const MODELS_DEV_CACHE_FILENAME = "litellm-models-dev.json";
const TOKEN_REFRESH_LEAD_MS = 5 * 60 * 1000;
const PERMANENT_TOKEN_EXPIRES_AT = Number.MAX_SAFE_INTEGER;
const EXPIRE_TOKEN_IMMEDIATELY = 0;

type RawProviderSettings = {
  displayName?: unknown;
  baseUrl?: unknown;
  apiKey?: unknown;
  headers?: unknown;
  enabled?: unknown;
};

type ProviderDefinition = {
  name: string;
  displayName: string;
  baseUrl?: string;
  apiKeyConfig?: string;
  headers?: unknown;
  useDefaultEnv: boolean;
  useGcloudTokenAuth: boolean;
  enableOAuth: boolean;
};

function getModelsDevDiscoveryOptions(): Pick<DiscoveryOptions, "modelsDev" | "modelsDevCachePath"> {
  return {
    modelsDev: process.env[ENV_MODELS_DEV] !== "0",
    modelsDevCachePath: join(getAgentDir(), MODELS_DEV_CACHE_FILENAME),
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readGlobalLiteLLMSettings(): Promise<Record<string, unknown> | undefined> {
  try {
    const raw = await readFile(join(getAgentDir(), "settings.json"), "utf8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const settings = parsed[SETTINGS_KEY];
    return settings && typeof settings === "object" && !Array.isArray(settings)
      ? (settings as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

function cleanConfig(raw: string | undefined): string | undefined {
  const trimmed = raw?.trim();
  return trimmed && trimmed !== "undefined" ? trimmed : undefined;
}

function resolveSanitizedCredentialRoot(
  definition: ProviderDefinition,
  credential?: Credential,
  requestBaseUrl?: string,
): string | undefined {
  const credentialBaseUrl =
    credential?.type === "oauth"
      ? cleanConfig(typeof credential.baseUrl === "string" ? credential.baseUrl : undefined)
      : credential?.type === "api_key"
        ? cleanConfig(credential.env?.[ENV_BASE_URL])
        : undefined;
  const baseUrl =
    credentialBaseUrl ??
    cleanConfig(requestBaseUrl) ??
    cleanConfig(definition.baseUrl) ??
    (definition.useDefaultEnv ? cleanConfig(process.env[ENV_BASE_URL]) : undefined);
  if (!baseUrl) return undefined;
  try {
    const normalized = normalizeBaseUrl(baseUrl);
    const parsed = new URL(normalized);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") throw new Error("unsupported protocol");
    return normalized;
  } catch {
    throw new Error("Invalid LiteLLM base URL; a network refresh with a valid URL is required");
  }
}

function requireNonPlaceholderCredentialRoot(root: string): string {
  if (isPlaceholderHost(new URL(root).host)) {
    throw new Error("LiteLLM credentials use a placeholder host; a network refresh with a real base URL is required");
  }
  return root;
}

function stringSetting(value: unknown): string | undefined {
  return typeof value === "string" ? cleanConfig(value) : undefined;
}

function normalizeCommand(raw: string | undefined): string | undefined {
  const trimmed = cleanConfig(raw);
  if (!trimmed) return undefined;
  return trimmed.startsWith("!") ? trimmed : `!${trimmed}`;
}

function getApiKeyHelperCommand(): string | undefined {
  return normalizeCommand(process.env[ENV_API_KEY_HELPER]);
}

function parseApiKeyCommand(commandConfig: string): string[] {
  const command = commandConfig.startsWith("!") ? commandConfig.slice(1) : commandConfig;
  if (/[\n\r;&|<>`$(){}*?~%^!]|\[|\]/.test(command))
    throw new Error("LiteLLM API key helper shell syntax is not supported");

  const parts: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;
  for (let index = 0; index < command.length; index += 1) {
    const character = command[index]!;
    const next = command[index + 1];
    if (character === "\\" && quote !== "'" && next && (next === "\\" || next === quote || /\s/.test(next))) {
      current += next;
      index += 1;
    } else if ((character === "'" || character === '"') && (!quote || quote === character)) {
      quote = quote ? undefined : character;
    } else if (!quote && /\s/.test(character)) {
      if (current) {
        parts.push(current);
        current = "";
      }
    } else {
      current += character;
    }
  }

  if (quote) throw new Error(`LiteLLM API key helper command has an unterminated quote: ${command}`);
  if (current) parts.push(current);
  if (parts.length === 0) throw new Error("LiteLLM API key helper command is empty");
  if (/\.(?:cmd|bat)$/i.test(parts[0]!))
    throw new Error("LiteLLM API key helper shell scripts are not supported; use an executable");
  return parts;
}

function executeApiKeyCommand(commandConfig: string): string {
  const [command, ...args] = parseApiKeyCommand(commandConfig);
  const output = execFileSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 10_000,
  }).trim();
  if (!output) throw new Error(`LiteLLM API key helper produced no output: ${command}`);
  return output;
}

function tokenExpiresAt(apiKey: string, opaqueFallback = PERMANENT_TOKEN_EXPIRES_AT): number {
  const [, payload] = apiKey.split(".");
  if (!payload) return opaqueFallback;
  try {
    const claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as { exp?: unknown };
    return typeof claims.exp === "number"
      ? Math.max(Date.now(), claims.exp * 1000 - TOKEN_REFRESH_LEAD_MS)
      : opaqueFallback;
  } catch {
    return opaqueFallback;
  }
}

async function generateVirtualKey(
  baseUrl: string,
  userToken: string,
  signal?: AbortSignal,
  headers?: Record<string, string>,
): Promise<{ key: string; expiresAt?: number }> {
  const boundedSignal = signal
    ? AbortSignal.any([signal, AbortSignal.timeout(LOGIN_TIMEOUT_MS)])
    : AbortSignal.timeout(LOGIN_TIMEOUT_MS);
  const response = await fetch(`${baseUrl}/key/generate`, {
    method: "POST",
    headers: {
      ...headers,
      Authorization: `Bearer ${userToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({}),
    signal: boundedSignal,
  });
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Virtual key generation failed (${response.status}): ${text}`);
  }
  const data = (await response.json()) as { key?: unknown; expires?: unknown };
  if (typeof data.key !== "string" || !data.key) throw new Error("No key in response from /key/generate");
  const expiresMs = typeof data.expires === "string" ? Date.parse(data.expires) : Number.NaN;
  return { key: data.key, expiresAt: Number.isNaN(expiresMs) ? undefined : expiresMs };
}

function resolveTemplateConfigValue(config: string): string | undefined {
  let resolved = "";
  for (let index = 0; index < config.length; ) {
    const dollarIndex = config.indexOf("$", index);
    if (dollarIndex === -1) return resolved + config.slice(index);
    resolved += config.slice(index, dollarIndex);
    const nextChar = config[dollarIndex + 1];
    if (nextChar === "$" || nextChar === "!") {
      resolved += nextChar;
      index = dollarIndex + 2;
      continue;
    }
    if (nextChar === "{") {
      const endIndex = config.indexOf("}", dollarIndex + 2);
      if (endIndex === -1) {
        resolved += "$";
        index = dollarIndex + 1;
        continue;
      }
      const name = config.slice(dollarIndex + 2, endIndex);
      const envValue = process.env[name];
      if (envValue === undefined) return undefined;
      resolved += envValue;
      index = endIndex + 1;
      continue;
    }
    const match = config.slice(dollarIndex + 1).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (!match) {
      resolved += "$";
      index = dollarIndex + 1;
      continue;
    }
    const envValue = process.env[match[0]];
    if (envValue === undefined) return undefined;
    resolved += envValue;
    index = dollarIndex + 1 + match[0].length;
  }
  return resolved;
}

async function resolveTemplateConfigValueFromContext(
  config: string,
  env: (name: string) => Promise<string | undefined>,
): Promise<string | undefined> {
  let resolved = "";
  for (let index = 0; index < config.length; ) {
    const dollarIndex = config.indexOf("$", index);
    if (dollarIndex === -1) return resolved + config.slice(index);
    resolved += config.slice(index, dollarIndex);
    const nextChar = config[dollarIndex + 1];
    if (nextChar === "$" || nextChar === "!") {
      resolved += nextChar;
      index = dollarIndex + 2;
      continue;
    }
    if (nextChar === "{") {
      const endIndex = config.indexOf("}", dollarIndex + 2);
      if (endIndex === -1) {
        resolved += "$";
        index = dollarIndex + 1;
        continue;
      }
      const envValue = await env(config.slice(dollarIndex + 2, endIndex));
      if (envValue === undefined) return undefined;
      resolved += envValue;
      index = endIndex + 1;
      continue;
    }
    const match = config.slice(dollarIndex + 1).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (!match) {
      resolved += "$";
      index = dollarIndex + 1;
      continue;
    }
    const envValue = await env(match[0]);
    if (envValue === undefined) return undefined;
    resolved += envValue;
    index = dollarIndex + 1 + match[0].length;
  }
  return resolved;
}

function resolveConfigValue(config: string, { executeCommands }: { executeCommands: boolean }): string | undefined {
  if (config.startsWith("!")) return executeCommands ? executeApiKeyCommand(config) : undefined;
  return resolveTemplateConfigValue(config);
}

const warnedUnresolvedApiKeys = new Set<string>();

function warnUnresolvedApiKeyConfig(providerName: string, config: string): void {
  const key = `${providerName} ${config}`;
  if (warnedUnresolvedApiKeys.has(key)) return;
  warnedUnresolvedApiKeys.add(key);
  process.stderr.write(
    `LiteLLM (${providerName}): configured apiKey did not resolve (unset environment variable?); use $$ for a literal $.\n`,
  );
}

function parseHeaderRecord(value: unknown): Record<string, string> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const headers: Record<string, string> = {};
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (!key.trim()) continue;
    let resolved: string | undefined;
    if (typeof raw === "string") resolved = resolveTemplateConfigValue(raw);
    else if (typeof raw === "number" || typeof raw === "boolean") resolved = String(raw);
    else {
      process.stderr.write(`LiteLLM: ignoring non-primitive header value for "${key}".\n`);
      continue;
    }
    if (resolved) headers[key] = resolved;
  }
  return Object.keys(headers).length > 0 ? headers : undefined;
}

function parseCustomHeaders(raw: string | undefined): Record<string, string> | undefined {
  const trimmed = cleanConfig(raw);
  if (!trimmed) return undefined;
  try {
    return parseHeaderRecord(JSON.parse(trimmed));
  } catch (error) {
    process.stderr.write(
      `LiteLLM: failed to parse custom headers (${error instanceof Error ? error.message : String(error)}).\n`,
    );
    return undefined;
  }
}

function resolveHeaders(definition: ProviderDefinition): Record<string, string> | undefined {
  if (typeof definition.headers === "string") return parseCustomHeaders(resolveTemplateConfigValue(definition.headers));
  return parseHeaderRecord(definition.headers);
}

async function resolveHeadersFromContext(
  definition: ProviderDefinition,
  env: (name: string) => Promise<string | undefined>,
): Promise<Record<string, string> | undefined> {
  if (typeof definition.headers === "string")
    return parseCustomHeaders(await resolveTemplateConfigValueFromContext(definition.headers, env));
  return parseHeaderRecord(definition.headers);
}

async function resolveCredentials(
  definition: ProviderDefinition,
  { executeHelpers = true } = {},
): Promise<ResolvedCredentials> {
  const configuredBase =
    cleanConfig(definition.baseUrl) ?? (definition.useDefaultEnv ? cleanConfig(process.env[ENV_BASE_URL]) : undefined);
  const envKey = definition.useDefaultEnv ? cleanConfig(process.env[ENV_API_KEY]) : undefined;
  const envHelperCommand = definition.useDefaultEnv ? getApiKeyHelperCommand() : undefined;
  const useGcloudToken = definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled();
  const gcloudKey = executeHelpers && useGcloudToken ? (await getGcloudToken())?.trim() : undefined;
  // Resolved lazily so a `!command` key is not executed when a
  // higher-precedence credential (saved auth, gcloud token) already won.
  let configuredKey: string | undefined;
  if (!gcloudKey && definition.apiKeyConfig) {
    configuredKey = resolveConfigValue(definition.apiKeyConfig, { executeCommands: executeHelpers });
    if (configuredKey === undefined && !definition.apiKeyConfig.startsWith("!")) {
      warnUnresolvedApiKeyConfig(definition.name, definition.apiKeyConfig);
    }
  }
  const helperKey =
    !gcloudKey && !configuredKey && executeHelpers && envHelperCommand
      ? executeApiKeyCommand(envHelperCommand)
      : undefined;
  const apiKey = gcloudKey || configuredKey || helperKey || envKey;

  let apiKeyConfig: string | undefined;
  if (configuredKey && definition.apiKeyConfig) {
    apiKeyConfig = definition.apiKeyConfig;
  } else if (!executeHelpers && definition.apiKeyConfig?.startsWith("!")) {
    apiKeyConfig = definition.apiKeyConfig;
  } else if (helperKey && envHelperCommand) {
    apiKeyConfig = envHelperCommand;
  } else if (!executeHelpers && envHelperCommand) {
    apiKeyConfig = envHelperCommand;
  } else if (envKey) {
    apiKeyConfig = `$${ENV_API_KEY}`;
  }
  return {
    baseUrl: configuredBase ? normalizeBaseUrl(configuredBase) : undefined,
    apiKey: apiKey || undefined,
    apiKeyConfig,
    apiKeyFromGcloudAdc: Boolean(gcloudKey),
  };
}

function getDiscoveryTimeoutMs(): number {
  const raw = process.env[ENV_TIMEOUT];
  if (raw === undefined) return DEFAULT_TIMEOUT_MS;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < 0) return DEFAULT_TIMEOUT_MS;
  return parsed;
}

function isOffline(): boolean {
  return process.env[ENV_OFFLINE] === "1";
}

function isVerboseDiscovery(): boolean {
  return process.env[ENV_VERBOSE_DISCOVERY] === "1";
}

function normalizeProviderSettings(raw: unknown): RawProviderSettings | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const record = raw as RawProviderSettings;
  if (record.enabled === false) return undefined;
  return record;
}

function isFeatureEnabled(settings: Record<string, unknown> | undefined, feature: "skills" | "mcp"): boolean {
  const raw = settings?.[feature];
  return !isPlainObject(raw) || raw.enabled !== false;
}

function getProviderDefinitions(settings: Record<string, unknown> | undefined): ProviderDefinition[] {
  const rawProviders = settings?.providers && typeof settings.providers === "object" ? settings.providers : undefined;
  const providerSettings = rawProviders as Record<string, unknown> | undefined;
  const defaultSettings = normalizeProviderSettings(providerSettings?.[PROVIDER_NAME]);

  const makeDefinition = (
    name: string,
    raw: RawProviderSettings | undefined,
    isDefault: boolean,
  ): ProviderDefinition => ({
    name,
    displayName: stringSetting(raw?.displayName) ?? (isDefault ? "LiteLLM" : name),
    baseUrl: stringSetting(raw?.baseUrl),
    apiKeyConfig: stringSetting(raw?.apiKey),
    headers: raw?.headers ?? (isDefault ? `$${ENV_HEADERS}` : undefined),
    useDefaultEnv: isDefault,
    useGcloudTokenAuth: isDefault,
    enableOAuth: isDefault,
  });

  const definitions = [makeDefinition(PROVIDER_NAME, defaultSettings, true)];
  for (const [name, raw] of Object.entries(providerSettings ?? {})) {
    if (name === PROVIDER_NAME) continue;
    const normalized = normalizeProviderSettings(raw);
    if (!normalized) continue;
    definitions.push(makeDefinition(name, normalized, false));
  }
  return definitions;
}

// Rejects a typed proxy URL at the prompt rather than storing it. A credential
// carrying a scheme-less or non-http(s) root fails on every later turn, far from
// the place the value was entered, so the clean error belongs here.
function requireLoginBaseUrl(rawBaseUrl: string): string {
  const normalized = normalizeBaseUrl(rawBaseUrl);
  let parsed: URL;
  try {
    parsed = new URL(normalized);
  } catch {
    throw new Error(
      `"${rawBaseUrl}" is not a valid URL; include the scheme, for example https://litellm.your-domain.com`,
    );
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    // `localhost:4000` parses as a `localhost:` scheme rather than failing above, so
    // this branch is what a missing scheme actually lands on.
    throw new Error(
      `"${rawBaseUrl}" must use http or https; include the scheme, for example https://litellm.your-domain.com`,
    );
  }
  if (isPlaceholderHost(parsed.host)) {
    throw new Error(`"${rawBaseUrl}" is the documentation placeholder; enter your own LiteLLM proxy URL`);
  }
  return normalized;
}

async function loginApiKey(interaction: AuthInteraction): Promise<ApiKeyCredential> {
  const rawBaseUrl = (
    await interaction.prompt({
      type: "text",
      message: "Enter LiteLLM proxy URL (no trailing /v1):",
      placeholder: "https://litellm.your-domain.com",
    })
  ).trim();
  if (!rawBaseUrl) throw new Error("Base URL is required");
  const baseUrl = requireLoginBaseUrl(rawBaseUrl);
  const key = (await interaction.prompt({ type: "secret", message: "Enter API key:" })).trim();
  if (!key) throw new Error("Both base URL and API key are required");
  return { type: "api_key", key, env: { [ENV_BASE_URL]: baseUrl } };
}

async function loginOAuth(interaction: AuthInteraction, headers?: Record<string, string>): Promise<OAuthCredential> {
  const rawBaseUrl = (
    await interaction.prompt({
      type: "text",
      message: "Enter LiteLLM proxy URL (no trailing /v1):",
      placeholder: "https://litellm.your-domain.com",
    })
  ).trim();
  if (!rawBaseUrl) throw new Error("Base URL is required");
  const baseUrl = requireLoginBaseUrl(rawBaseUrl);
  interaction.notify({
    type: "auth_url",
    url: `${baseUrl}/sso/key/generate`,
    instructions: "Authenticate via SSO, then copy your token from the LiteLLM UI.",
  });
  const rawToken = (await interaction.prompt({ type: "secret", message: "Paste your SSO token from the LiteLLM UI:" }))
    .trim()
    .replace(/^Bearer\s+/i, "")
    .trim();
  if (!rawToken) throw new Error("SSO token is required");
  const wantVirtualKey = (
    await interaction.prompt({ type: "text", message: "Generate a LiteLLM virtual key from this token? (y/n):" })
  )
    .trim()
    .toLowerCase();
  let access = rawToken;
  let expires = tokenExpiresAt(rawToken, PERMANENT_TOKEN_EXPIRES_AT);
  if (wantVirtualKey !== "n" && wantVirtualKey !== "no") {
    try {
      interaction.notify({ type: "progress", message: "Generating virtual key..." });
      const generated = await generateVirtualKey(baseUrl, rawToken, interaction.signal, headers);
      access = generated.key;
      expires =
        generated.expiresAt === undefined
          ? PERMANENT_TOKEN_EXPIRES_AT
          : Math.max(Date.now(), generated.expiresAt - TOKEN_REFRESH_LEAD_MS);
      interaction.notify({ type: "progress", message: "Virtual key generated and will be used for API calls." });
    } catch (error) {
      if (interaction.signal?.aborted) throw interaction.signal.reason;
      const message = error instanceof Error ? error.message : String(error);
      interaction.notify({
        type: "progress",
        message: `LiteLLM: virtual key generation failed (${message}); using SSO token directly.`,
      });
    }
  }
  return { type: "oauth", access, refresh: "", expires, baseUrl };
}

async function refreshLiteLLM(credentials: OAuthCredentials, _signal?: AbortSignal): Promise<OAuthCredentials> {
  if (!credentials.refresh.startsWith("!")) {
    if (credentials.expires < PERMANENT_TOKEN_EXPIRES_AT) {
      throw new Error("LiteLLM credential cannot be refreshed; run /login litellm again");
    }
    return credentials;
  }
  const access = executeApiKeyCommand(credentials.refresh);
  return { ...credentials, access, expires: tokenExpiresAt(access, EXPIRE_TOKEN_IMMEDIATELY) };
}

async function resolveApiKeyAuth(
  definition: ProviderDefinition,
  ctx: { env(name: string): Promise<string | undefined> },
  credential?: ApiKeyCredential,
  executeHelpers = true,
) {
  const baseUrl =
    cleanConfig(credential?.env?.[ENV_BASE_URL]) ??
    cleanConfig(definition.baseUrl) ??
    (definition.useDefaultEnv ? cleanConfig(await ctx.env(ENV_BASE_URL)) : undefined);
  const stored = credential?.key
    ? resolveConfigValue(credential.key, { executeCommands: executeHelpers })?.trim()
    : undefined;
  let source: string | undefined;
  let creds: ResolvedCredentials;
  if (stored) {
    source = "stored credential";
    creds = {
      baseUrl: baseUrl ? normalizeBaseUrl(baseUrl) : undefined,
      apiKey: stored,
    };
  } else {
    creds = await resolveCredentials(
      { ...definition, apiKeyConfig: undefined, useDefaultEnv: false },
      { executeHelpers },
    );
    if (!creds.apiKey && definition.apiKeyConfig) {
      const configured = definition.apiKeyConfig.startsWith("!")
        ? executeHelpers
          ? executeApiKeyCommand(definition.apiKeyConfig)
          : undefined
        : await resolveTemplateConfigValueFromContext(definition.apiKeyConfig, ctx.env);
      if (configured) {
        creds.apiKey = configured;
        creds.apiKeyConfig = definition.apiKeyConfig;
      } else if (!definition.apiKeyConfig.startsWith("!")) {
        warnUnresolvedApiKeyConfig(definition.name, definition.apiKeyConfig);
      }
    }
    if (!creds.apiKey && definition.useDefaultEnv) {
      const helper = normalizeCommand(await ctx.env(ENV_API_KEY_HELPER));
      if (helper) {
        creds.apiKey = executeHelpers ? executeApiKeyCommand(helper) : undefined;
        creds.apiKeyConfig = helper;
        source = ENV_API_KEY_HELPER;
      } else {
        const envKey = cleanConfig(await ctx.env(ENV_API_KEY));
        if (envKey) {
          creds.apiKey = envKey;
          creds.apiKeyConfig = ENV_API_KEY;
          source = ENV_API_KEY;
        }
      }
    }
    if (!creds.baseUrl && baseUrl) creds.baseUrl = normalizeBaseUrl(baseUrl);
  }
  if (!creds.apiKey) return undefined;
  return {
    auth: {
      apiKey: creds.apiKey,
      headers: await resolveHeadersFromContext(definition, ctx.env),
    },
    env: baseUrl ? { [ENV_BASE_URL]: normalizeBaseUrl(baseUrl) } : undefined,
    source: source ?? (creds.apiKeyFromGcloudAdc ? GCLOUD_ADC_SOURCE : undefined) ?? creds.apiKeyConfig ?? ENV_API_KEY,
  };
}

function createProviderAuth(
  definition: ProviderDefinition,
  rememberOAuthRuntimeRoot: (apiKey: string, root: string) => void,
): ProviderAuth {
  return {
    apiKey: {
      name: `${definition.displayName} API key`,
      login: definition.name === PROVIDER_NAME ? loginApiKey : undefined,
      check: async ({ ctx, credential }) => {
        const baseUrl =
          credential?.env?.[ENV_BASE_URL] ??
          definition.baseUrl ??
          (definition.useDefaultEnv ? await ctx.env(ENV_BASE_URL) : undefined);
        if (!cleanConfig(baseUrl)) return undefined;
        if (credential?.key) return { type: "api_key", source: "stored credential" };

        // Credentials `resolve` would fall back to if ADC cannot mint a token.
        const fallbackSource = async (): Promise<string | undefined> => {
          if (definition.apiKeyConfig) {
            const configuredKey = definition.apiKeyConfig.startsWith("!")
              ? definition.apiKeyConfig
              : await resolveTemplateConfigValueFromContext(definition.apiKeyConfig, ctx.env);
            if (configuredKey) return definition.apiKeyConfig;
          }
          if (definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY_HELPER))) return ENV_API_KEY_HELPER;
          if (definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY))) return ENV_API_KEY;
          return undefined;
        };

        // Mirror the precedence in `resolveCredentials`, where ADC outranks the config
        // key, the helper and the environment key. Whether the refresh token still mints
        // is only knowable at request time, and this must not make a network call; if it
        // fails, `resolve` falls back and reports the credential it actually used.
        if (definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled() && (await hasGcloudAdcCredentials()))
          return { type: "api_key", source: GCLOUD_ADC_SOURCE };
        const fallback = await fallbackSource();
        return fallback ? { type: "api_key", source: fallback } : undefined;
      },
      resolve: ({ ctx, credential }) => resolveApiKeyAuth(definition, ctx, credential),
    },
    oauth: definition.enableOAuth
      ? {
          name: "LiteLLM SSO",
          loginLabel: "Sign in with LiteLLM SSO",
          login: (interaction) => loginOAuth(interaction, resolveHeaders(definition)),
          refresh: async (credential, signal) => ({
            ...(await refreshLiteLLM(credential, signal)),
            type: "oauth" as const,
          }),
          // Deliberately returns no baseUrl. Pi overwrites model.baseUrl with
          // auth.baseUrl before dispatch, which would replace a stale cached host with
          // the credential root and make the host comparison in provider.ts compare the
          // credential against itself -- silently repointing a stale model instead of
          // rejecting it. The root is still resolved here so an unusable credential
          // fails at login/refresh time, and it is remembered against this access token
          // for the guarded request path and the Skills/MCP surfaces to reproject from.
          toAuth: async (credential) => {
            const baseUrl = resolveSanitizedCredentialRoot(definition, credential);
            if (!baseUrl) throw new Error("LiteLLM OAuth credential has no valid base URL; run /login litellm again");
            rememberOAuthRuntimeRoot(credential.access, baseUrl);
            return {
              apiKey: credential.access,
              headers: resolveHeaders(definition),
            };
          },
        }
      : undefined,
  };
}

function isReasoningItem(item: unknown): boolean {
  return typeof item === "object" && item !== null && (item as { type?: unknown }).type === "reasoning";
}

// Returns the original array reference when no normalization is needed.
function normalizeStrictToolMessages(messages: any[]): any[] {
  const normalized: any[] = [];
  let imageParts: any[] = [];
  let changed = false;
  const flushImages = (): void => {
    if (imageParts.length === 0) return;
    normalized.push({
      role: "user",
      content: [{ type: "text", text: "Attached image(s) from tool result:" }, ...imageParts],
    });
    imageParts = [];
  };

  for (const message of messages) {
    if (message.role !== "tool") flushImages();
    if (
      message.role === "assistant" &&
      Array.isArray(message.tool_calls) &&
      message.tool_calls.length > 0 &&
      message.content == null
    ) {
      normalized.push({ ...message, content: "" });
      changed = true;
    } else if (message.role === "tool" && Array.isArray(message.content)) {
      const previousImageCount = imageParts.length;
      const content = message.content
        .map((part: any) => {
          if (typeof part === "string") return part;
          if (part?.type === "text") return part.text || "";
          if (part?.type === "image_url") imageParts.push(part);
          return "";
        })
        .join("");
      normalized.push({
        ...message,
        content: content || (imageParts.length > previousImageCount ? "(see attached image)" : "(no tool output)"),
      });
      changed = true;
    } else {
      normalized.push(message);
    }
  }
  flushImages();
  return changed ? normalized : messages;
}

// Reasoning fields LiteLLM forwards to chat-completions providers. The Moonshot
// path defaults them off; the gpt-5.5 tool path strips them entirely.
const REASONING_SUPPRESSION_DEFAULTS: Record<string, unknown> = {
  include_reasoning: false,
  reasoning_content: false,
  merge_reasoning_content_in_choices: true,
  thinking: { type: "disabled" },
};
const GEMINI_MODEL_PATTERN = /(?:^|[-_/.:])gemini(?=$|[-_/.:])/i;

function prepareLiteLLMRequestPayload(
  payload: Record<string, unknown>,
  modelId: string | undefined,
  api: Api | undefined,
  sessionId: string | undefined,
  modelPolicy: LiteLLMModelPolicy | undefined,
): Record<string, unknown> | undefined {
  const openAIApi = api ?? "openai-completions";
  let next: Record<string, unknown> | undefined;
  // LiteLLM still routes gpt-5.5 tool+reasoning requests through chat completions.
  // Drop reasoning until the gateway honors /v1/responses for this route.
  if (
    openAIApi === "openai-completions" &&
    modelId &&
    isGpt55Model(modelId) &&
    Array.isArray(payload.tools) &&
    payload.tools.length > 0
  ) {
    const reasoningKeys = ["reasoning", "reasoning_effort", ...Object.keys(REASONING_SUPPRESSION_DEFAULTS)];
    for (const key of reasoningKeys) {
      if (payload[key] === undefined) continue;
      next ??= { ...payload };
      delete next[key];
    }
    const include = (next ?? payload).include;
    if (Array.isArray(include) && include.includes("reasoning.encrypted_content")) {
      next ??= { ...payload };
      const filteredInclude = include.filter((value) => value !== "reasoning.encrypted_content");
      if (filteredInclude.length === 0) delete next.include;
      else next.include = filteredInclude;
    }
    // Prior turns may have replayed reasoning items (with encrypted_content)
    // into the input; they are rejected once reasoning is stripped.
    const input = (next ?? payload).input;
    if (Array.isArray(input) && input.some(isReasoningItem)) {
      next ??= { ...payload };
      next.input = input.filter((item) => !isReasoningItem(item));
    }
  }

  // Moonshot/Kimi applies strict OpenAI schema validation: assistant tool calls
  // must carry string content, and tool results must be plain text.
  if (openAIApi === "openai-completions" && modelId && modelPolicy?.normalizeStrictToolMessages) {
    const messages = (next ?? payload).messages;
    if (Array.isArray(messages)) {
      const normalized = normalizeStrictToolMessages(messages);
      if (normalized !== messages) {
        next ??= { ...payload };
        next.messages = normalized;
      }
    }
  }

  if (
    (openAIApi === "openai-completions" || openAIApi === "openai-responses") &&
    modelId &&
    GEMINI_MODEL_PATTERN.test(modelId)
  ) {
    const currentPayload = next ?? payload;
    if (typeof currentPayload.reasoning_effort === "string") {
      const lower = currentPayload.reasoning_effort.toLowerCase();
      if (currentPayload.reasoning_effort !== lower) {
        next ??= { ...payload };
        next.reasoning_effort = lower;
      }
    }

    const reasoningPayload = next ?? payload;
    if (isPlainObject(reasoningPayload.reasoning) && typeof reasoningPayload.reasoning.effort === "string") {
      const lower = reasoningPayload.reasoning.effort.toLowerCase();
      if (reasoningPayload.reasoning.effort !== lower) {
        next ??= { ...payload };
        next.reasoning = {
          ...(next.reasoning as Record<string, unknown>),
          effort: lower,
        };
      }
    }
  }

  if ((openAIApi === "openai-completions" || openAIApi === "openai-responses") && sessionId) {
    next ??= { ...payload };
    next.litellm_session_id = sessionId;
  }

  return next;
}

function normalizeThinkTags(
  message: AssistantMessage,
  litellmProviderNames: Set<string>,
  modelPolicy: LiteLLMModelPolicy | undefined,
): AssistantMessage | undefined {
  if (!litellmProviderNames.has(message.provider) || !modelPolicy?.normalizeThinkTags) return;

  let changed = false;
  const content: AssistantMessage["content"] = [];
  const appendText = (text: string): void => {
    if (!text) return;
    const last = content.at(-1);
    if (last?.type === "text") {
      last.text += text;
      return;
    }
    content.push({ type: "text", text });
  };
  const appendThinking = (thinking: string): void => {
    if (!thinking) return;
    const last = content.at(-1);
    if (last?.type === "thinking") {
      last.thinking += thinking;
      return;
    }
    content.push({ type: "thinking", thinking });
  };

  for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
    const block = message.content[blockIndex];
    if (block.type !== "text") {
      content.push(block);
      continue;
    }

    let index = 0;
    while (index < block.text.length) {
      const start = block.text.indexOf("<think>", index);
      if (start === -1) {
        appendText(block.text.slice(index));
        break;
      }

      changed = true;
      appendText(block.text.slice(index, start));
      const thinkingStart = start + "<think>".length;
      const end = block.text.indexOf("</think>", thinkingStart);
      if (end === -1) {
        const isBeforeNonTextContent = message.content
          .slice(blockIndex + 1)
          .some((nextBlock) => nextBlock.type !== "text");
        if (isBeforeNonTextContent) appendThinking(block.text.slice(thinkingStart));
        else appendText(block.text.slice(thinkingStart));
        index = block.text.length;
        break;
      }

      appendThinking(block.text.slice(thinkingStart, end));
      index = end + "</think>".length;
    }
  }

  if (!changed) return;
  return { ...message, content };
}

export default async function (pi: ExtensionAPI): Promise<void> {
  const settings = await readGlobalLiteLLMSettings();
  const definitions = getProviderDefinitions(settings);
  const skillsEnabled = isFeatureEnabled(settings, "skills");
  const mcpEnabled = isFeatureEnabled(settings, "mcp");
  const providerNames = new Set(definitions.map((definition) => definition.name));

  function discoveryDisabledReason(): string | null {
    if (isOffline()) return `${ENV_OFFLINE}=1`;
    if (getDiscoveryTimeoutMs() === 0) return `${ENV_TIMEOUT}=0`;
    return null;
  }

  function requestBaseUrl(definition: ProviderDefinition): string {
    try {
      return `${resolveSanitizedCredentialRoot(definition) ?? normalizeBaseUrl(DEFAULT_LITELLM_BASE_URL)}/v1`;
    } catch {
      return `${normalizeBaseUrl(DEFAULT_LITELLM_BASE_URL)}/v1`;
    }
  }

  async function authForCredential(definition: ProviderDefinition, credential: Credential) {
    if (credential.type === "oauth") {
      const baseUrl = resolveSanitizedCredentialRoot(definition, credential);
      if (!baseUrl) throw new Error(`no LiteLLM base URL for ${definition.name}. Run /login litellm again.`);
      return {
        baseUrl: requireNonPlaceholderCredentialRoot(baseUrl),
        apiKey: credential.access,
        headers: resolveHeaders(definition),
      };
    }
    const resolved = await resolveApiKeyAuth(definition, { env: async (name) => process.env[name] }, credential);
    if (!resolved?.auth.apiKey) {
      throw new Error(`no credentials for ${definition.name}. Run /login litellm or set env vars.`);
    }
    const baseUrl = resolveSanitizedCredentialRoot(definition, credential, resolved.env?.[ENV_BASE_URL]);
    if (!baseUrl) throw new Error(`no LiteLLM base URL for ${definition.name}. Run /login litellm or set env vars.`);
    return {
      baseUrl: requireNonPlaceholderCredentialRoot(baseUrl),
      apiKey: resolved.auth.apiKey,
      headers: resolved.auth.headers,
    };
  }

  let registeredMcpIdentity: string | undefined;
  let mcpRegistration: Promise<void> | undefined;
  let defaultRuntimeAuth: LiteLLMRuntimeAuth | undefined;
  // The host an OAuth login resolved, bound to the access token it was resolved for.
  // Request paths and the Skills/MCP surfaces receive no Credential, and OAuth
  // credentials contribute no request env, so this is the only credential-derived
  // source of a root for them. Token-keyed so a different credential cannot read it,
  // and cleared when a non-OAuth credential resolves so it cannot outlive /logout.
  // Only the default provider enables OAuth, so one slot is sufficient.
  let oauthRuntimeRoot: { apiKey: string; root: string } | undefined;
  const rememberOAuthRuntimeRoot = (apiKey: string, root: string): void => {
    oauthRuntimeRoot = { apiKey, root };
  };
  const oauthRuntimeRootFor = (apiKey: string | undefined): string | undefined =>
    apiKey && oauthRuntimeRoot?.apiKey === apiKey ? oauthRuntimeRoot.root : undefined;
  const reportedRuntimeAuthFailures = new Set<string>();
  const reportRuntimeAuthOnce = (message: string): void => {
    if (reportedRuntimeAuthFailures.has(message)) return;
    reportedRuntimeAuthFailures.add(message);
    process.stderr.write(`LiteLLM (${PROVIDER_NAME}): ${message}\n`);
  };

  function mcpCatalogIdentity(auth: LiteLLMRuntimeAuth): string {
    return JSON.stringify({
      baseUrl: normalizeBaseUrl(auth.baseUrl),
      apiKey: auth.apiKey,
      headers: Object.entries(auth.headers ?? {}).sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0)),
    });
  }

  // Returns undefined rather than throwing when the configured base URL is absent,
  // malformed, or still the placeholder. This runs from before_agent_start on every
  // turn, including turns that never touch LiteLLM, so a configuration problem here
  // must not surface as a per-turn extension error. filterModels already reports the
  // reason once per session, and the LiteLLM tool surfaces raise their own error via
  // resolveDefaultRuntimeAuth.
  // Containment wraps the ENTIRE derivation, including getProviderAuth. Pi resolves
  // credentials inside that call, which runs our own oauth.toAuth, so a stored
  // credential with an invalid root throws from there rather than from the validator
  // we call directly. This runs from before_agent_start on every turn, including
  // turns that never touch LiteLLM, so nothing here may escape: filterModels owns
  // the model-list explanation and resolveDefaultRuntimeAuth owns the tool-surface
  // error. The reason is reported once so a real misconfiguration is not silent.
  async function getRuntimeAuth(ctx: ExtensionContext): Promise<LiteLLMRuntimeAuth | undefined> {
    try {
      const auth = await ctx.modelRegistry.getProviderAuth(PROVIDER_NAME);
      if (!auth) return undefined;
      const apiKey = auth.auth.apiKey;
      if (!apiKey) return undefined;
      const provider = ctx.modelRegistry.getProvider(PROVIDER_NAME);
      const root = resolveSanitizedCredentialRoot(
        definitions[0],
        undefined,
        auth.auth.baseUrl ?? auth.env?.[ENV_BASE_URL] ?? oauthRuntimeRootFor(apiKey),
      );
      if (!root || isPlaceholderHost(new URL(root).host)) return undefined;
      const headers = Object.fromEntries(
        Object.entries(auth.auth.headers ?? provider?.headers ?? {}).filter(
          (entry): entry is [string, string] => typeof entry[1] === "string",
        ),
      );
      return {
        baseUrl: normalizeBaseUrl(root),
        apiKey,
        headers: Object.keys(headers).length > 0 ? headers : undefined,
      };
    } catch (error) {
      reportRuntimeAuthOnce(error instanceof Error ? error.message : String(error));
      return undefined;
    }
  }

  async function resolveDefaultRuntimeAuth(ctx?: ExtensionContext): Promise<LiteLLMRuntimeAuth> {
    if (ctx?.modelRegistry) {
      const auth = await getRuntimeAuth(ctx);
      if (auth) return auth;
      throw new Error("no credentials for litellm. Run /login litellm or set env vars.");
    }
    if (!defaultRuntimeAuth) throw new Error("no credentials for litellm. Run /login litellm or set env vars.");
    return defaultRuntimeAuth;
  }

  function waitForMcpRegistration(registration: Promise<void>, signal?: AbortSignal): Promise<void> {
    if (!signal) return registration.catch(() => undefined);
    signal.throwIfAborted();
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        signal.removeEventListener("abort", onAbort);
        reject(signal.reason);
      };
      const onComplete = () => {
        signal.removeEventListener("abort", onAbort);
        resolve();
      };
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) onAbort();
      else registration.then(onComplete, onComplete);
    });
  }

  async function registerMcpTools(auth: LiteLLMRuntimeAuth, signal?: AbortSignal): Promise<void> {
    if (!mcpEnabled || discoveryDisabledReason()) return;
    const identity = mcpCatalogIdentity(auth);
    while (mcpRegistration) {
      await waitForMcpRegistration(mcpRegistration, signal);
      signal?.throwIfAborted();
      if (registeredMcpIdentity === identity) return;
    }
    if (registeredMcpIdentity === identity) return;

    const registration = (async () => {
      try {
        signal?.throwIfAborted();
        const tools = await createMcpToolDefinitions(
          (ctx) => (ctx?.modelRegistry ? resolveDefaultRuntimeAuth(ctx) : Promise.resolve(auth)),
          isVerboseDiscovery() ? (message) => process.stderr.write(`LiteLLM MCP: ${message}\n`) : undefined,
          signal,
        );
        signal?.throwIfAborted();
        for (const tool of tools) pi.registerTool(tool);
        registeredMcpIdentity = identity;
      } catch (error) {
        if (signal?.aborted) throw signal.reason;
        process.stderr.write(
          `LiteLLM (${PROVIDER_NAME}): MCP tool discovery failed (${error instanceof Error ? error.message : String(error)}).\n`,
        );
      }
    })();
    mcpRegistration = registration;
    try {
      await registration;
    } finally {
      if (mcpRegistration === registration) mcpRegistration = undefined;
    }
  }

  for (const definition of definitions) {
    const providerAuth = createProviderAuth(definition, rememberOAuthRuntimeRoot);
    const apiKeyAuth = providerAuth.apiKey;
    if (apiKeyAuth) {
      const apiKeyResolve = apiKeyAuth.resolve;
      apiKeyAuth.resolve = async (input) => {
        // Resolving an api key means OAuth is no longer the active credential, so the
        // remembered SSO root must not survive the transition.
        oauthRuntimeRoot = undefined;
        return apiKeyResolve(input);
      };
    }
    const provider = createLiteLLMProvider({
      id: definition.name,
      name: definition.displayName,
      baseUrl: requestBaseUrl(definition),
      auth: providerAuth,
      // An explicit per-request base URL always wins; the remembered SSO root only
      // fills the gap where OAuth supplies no request env, ahead of configured and
      // environment defaults. Ordering the memo first made a stale root outrank the
      // live credential and wedged the provider until restart.
      resolveCredentialRoot: ({ credential, requestBaseUrl, apiKey }) =>
        resolveSanitizedCredentialRoot(definition, credential, requestBaseUrl ?? oauthRuntimeRootFor(apiKey)),
      discover: async (credential, signal) => {
        const disabledReason = discoveryDisabledReason();
        if (disabledReason) throw new Error(`discovery disabled (${disabledReason})`);
        const auth = await authForCredential(definition, credential);
        const result = await discoverModels(auth.baseUrl, auth.apiKey, {
          ...getModelsDevDiscoveryOptions(),
          timeoutMs: getDiscoveryTimeoutMs(),
          signal,
          headers: auth.headers,
          silent: !isVerboseDiscovery(),
          onProgress: isVerboseDiscovery() ? (message) => process.stderr.write(`LiteLLM: ${message}\n`) : undefined,
        });
        signal?.throwIfAborted();
        return { ...result, baseUrl: `${normalizeBaseUrl(auth.baseUrl)}/v1` };
      },
    });
    Object.assign(provider, { headers: resolveHeaders(definition) });
    const refreshModels = provider.refreshModels!;
    provider.refreshModels = async (context) => {
      try {
        await refreshModels(context);
      } finally {
        if (
          definition.name === PROVIDER_NAME &&
          context.allowNetwork &&
          !discoveryDisabledReason() &&
          context.credential
        ) {
          const auth = await authForCredential(definition, context.credential);
          defaultRuntimeAuth = auth;
          const registration = registerMcpTools(auth, context.signal);
          if (context.signal) await registration;
          else void registration.catch(() => undefined);
        }
      }
    };
    pi.registerProvider(provider);
  }

  setupLiteLLMCostTracking(pi, [...providerNames]);

  if (skillsEnabled) {
    for (const tool of createSkillToolDefinitions(resolveDefaultRuntimeAuth)) {
      pi.registerTool(tool);
    }
  }

  let sessionId: string | undefined;
  pi.on("session_start", (_event, ctx) => {
    sessionId = getSessionIdFromFile(ctx.sessionManager.getSessionFile());
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (!ctx.model?.provider || !providerNames.has(ctx.model.provider)) return;
    if (typeof event.payload !== "object" || event.payload === null) return;
    return prepareLiteLLMRequestPayload(
      event.payload as Record<string, unknown>,
      ctx.model?.id,
      ctx.model?.api,
      sessionId,
    );
  });

  pi.on("before_agent_start", async (event, ctx) => {
    defaultRuntimeAuth = undefined;
    if (!skillsEnabled || discoveryDisabledReason()) return;
    const auth = await getRuntimeAuth(ctx);
    if (!auth) return;
    defaultRuntimeAuth = auth;
    const skills = await listSkills(defaultRuntimeAuth.baseUrl, defaultRuntimeAuth.apiKey, defaultRuntimeAuth.headers);
    const section = createSkillsPromptSection(skills);
    if (!section) return;
    return { systemPrompt: `${event.systemPrompt}\n\n${section}` };
  });

  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    const model = ctx.modelRegistry?.find(event.message.provider, event.message.model);
    const modelPolicy = (model as (typeof model & { litellmPolicy?: LiteLLMModelPolicy }) | undefined)?.litellmPolicy;
    const message = normalizeThinkTags(event.message as AssistantMessage, providerNames, modelPolicy);
    if (!message) return;
    return { message };
  });
}
