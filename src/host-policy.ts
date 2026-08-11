import { normalizeBaseUrl } from "./discover.js";

// Sample host used when no LiteLLM base URL is configured. It is a documentation
// placeholder, never a real proxy, so credentials must never be sent to it.
export const DEFAULT_LITELLM_BASE_URL = "https://litellm.example.com";

// Compared on hostname only: a port or a fully-qualified trailing dot still reaches the
// same third-party documentation host, so neither may be used to slip past this gate.
const PLACEHOLDER_HOSTNAMES = new Set([canonicalHostname(new URL(DEFAULT_LITELLM_BASE_URL).hostname)]);

function canonicalHostname(hostname: string): string {
  return hostname.toLowerCase().replace(/\.$/, "");
}

// Every LiteLLM host rule lives here so availability, request, and discovery paths cannot drift apart.
export function isPlaceholderHost(host: string): boolean {
  // Accepts either a bare hostname or a `host:port` value.
  const hostname = host.startsWith("[") ? (host.match(/^\[[^\]]*\]/)?.[0] ?? host) : (host.split(":")[0] ?? host);
  return PLACEHOLDER_HOSTNAMES.has(canonicalHostname(hostname));
}

export function refreshRequired(message: string): Error {
  return new Error(`${message}; a network refresh with a valid LiteLLM base URL is required`);
}

// Normalize a LiteLLM root and return it with its comparable host. Throws when the
// value is not an absolute http(s) URL; callers that must not fail a turn should use
// credentialRootResult.
export function credentialRoot(baseUrl: string, subject: string): { root: string; host: string } {
  try {
    const root = normalizeBaseUrl(baseUrl);
    const url = new URL(root);
    if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("unsupported protocol");
    // Host identity is compared as canonical hostname plus port, so a trailing dot
    // cannot present one host as two and split cached models from active credentials.
    const port = url.port ? `:${url.port}` : "";
    return { root, host: `${canonicalHostname(url.hostname)}${port}` };
  } catch {
    throw refreshRequired(`${subject} has an invalid LiteLLM model URL`);
  }
}

export function credentialRootResult(
  baseUrl: string,
  subject: string,
): { root: string; host: string } | { error: Error } {
  try {
    return credentialRoot(baseUrl, subject);
  } catch (error) {
    return { error: error as Error };
  }
}

// A usable root: syntactically valid and not the documentation placeholder.
export function activeCredentialRoot(baseUrl: string, subject = "Active credentials"): { root: string; host: string } {
  const resolved = credentialRoot(baseUrl, subject);
  if (isPlaceholderHost(resolved.host)) {
    throw refreshRequired(`${subject} use a placeholder LiteLLM model host`);
  }
  return resolved;
}
