# pi-provider-litellm

LiteLLM proxy native Provider extension for [Pi](https://pi.dev). Pi 0.81.0+ is required.

Discovers models from self-hosted LiteLLM proxies and registers them under Pi providers. The default provider is `litellm`; optional aliases can register additional LiteLLM providers with separate credentials. Supports `/login litellm`, LiteLLM MCP tools, LiteLLM Skills Gateway prompt injection, and Google ADC token auth. Tries `/model/info` first (admin endpoint with rich metadata), falls back to `/v1/models` (OpenAI-compatible) on 401/403/404, then tries `/health` plus per-endpoint `/model/info` for older LiteLLM proxies.

## Install

```bash
pi install npm:pi-provider-litellm
```

Pi fetches the package from npm and registers it. Add `-l` to install into project settings (`.pi/settings.json`) instead of global.

To try it without installing (one-off, current run only):

```bash
pi -e npm:pi-provider-litellm
```

<details>
<summary>Alternative: install from source</summary>

```bash
git clone https://github.com/balcsida/pi-provider-litellm.git ~/.pi/agent/extensions/pi-provider-litellm
```

Pi loads the TypeScript source entrypoint declared in `package.json` `pi.extensions`, so a source install needs no
build step and no `node_modules`. Install dependencies only to run the test suite or the local checks.

</details>

## Configure

### Option A — interactive login

Inside pi:

```
/login litellm
```

To configure an API key, run `/login`, choose `Sign in with an API key`, then choose `LiteLLM API key`. With `/login litellm`, choose `Sign in with an API key` directly.

You'll be prompted for the base URL and API key. Credentials are persisted to `~/.pi/agent/auth.json`.

#### Enterprise SSO login

If your LiteLLM proxy requires SSO/OAuth authentication (enterprise deployments), you can authenticate via a browser SSO flow and optionally pair the resulting JWT with a stable virtual key:

1. Run `/login litellm` inside pi and select `Sign in with LiteLLM SSO`
2. Enter the proxy URL
3. Your default browser opens the LiteLLM SSO login URL (e.g. `https://litellm.your-domain.com/sso/key/generate`) automatically — the URL is also displayed in case it can't be opened. Authenticate via SSO
4. Copy your token from the LiteLLM UI and paste it at the prompt (copying a full `Bearer ...` header value is fine — the prefix is stripped automatically)
5. When prompted to generate a virtual key, press Enter to accept (recommended) or enter `n` to use the JWT directly

When you generate a virtual key, the resulting `sk-...` key is stored as your credential and used for all API requests. If the proxy's key policy attaches an expiry to the generated key, Pi will prompt you to re-authenticate when it nears expiry; otherwise the key is treated as permanent until revoked in LiteLLM.

When using a JWT directly, the extension reads its `exp` claim and Pi will prompt you to re-authenticate when the token nears expiry. Run `/login litellm` again to refresh.

### Option B — environment variables

```bash
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_API_KEY="sk-..."
```

Stored pi credentials for `litellm` take precedence over `LITELLM_API_KEY`; the environment key is used when no saved credential exists. `LITELLM_BASE_URL` is used when no saved login base URL exists.

### Multiple LiteLLM provider aliases

Add alias providers in `~/.pi/agent/settings.json` under `litellm.providers`. Each alias is registered as a separate Pi provider name, so models appear as `litellm/model-id` and `litellm-anthropic/model-id`.

```json
{
  "litellm": {
    "providers": {
      "litellm-anthropic": {
        "baseUrl": "https://litellm.your-domain.com",
        "apiKey": "$LITELLM_CLAUDE_KEY",
        "headers": "$LITELLM_HEADERS"
      }
    }
  }
}
```

You can also override the default provider through the same shape:

```json
{
  "litellm": {
    "providers": {
      "litellm": {
        "baseUrl": "https://litellm.your-domain.com",
        "apiKey": "$LITELLM_API_KEY",
        "headers": "$LITELLM_HEADERS"
      },
      "litellm-anthropic": {
        "baseUrl": "https://litellm.your-domain.com",
        "apiKey": "$LITELLM_CLAUDE_KEY",
        "headers": "$LITELLM_HEADERS"
      }
    }
  }
}
```

Provider fields:

| Field | Default | Effect |
|---|---|---|
| `baseUrl` | `LITELLM_BASE_URL` for `litellm`; required for aliases | LiteLLM proxy URL, with or without `/v1`. Must be a full `http`/`https` URL; a provider with no resolvable base URL exposes no models (see [Model host enforcement](#model-host-enforcement)) |
| `apiKey` | `LITELLM_API_KEY_HELPER`/`LITELLM_API_KEY` for `litellm`; required for aliases | Pi config value for this provider's key. Use `$ENV_VAR`, `${ENV_VAR}`, `!command`, or a literal key. Escape a literal `$` as `$$`. |
| `headers` | `$LITELLM_HEADERS` for `litellm`; unset for aliases | JSON string env reference or inline object of request headers |
| `displayName` | provider name | Label shown in Pi UI |
| `enabled` | `true` | Set `false` to skip an alias |

`/login litellm` and Google ADC token auth remain scoped to the default `litellm` provider. Aliases use their configured `apiKey` or manually stored auth entries matching the alias name.

### Optional LiteLLM features

LiteLLM Skills and MCP integration are enabled by default. Disable either feature globally in `~/.pi/agent/settings.json`:

```json
{
  "litellm": {
    "skills": {
      "enabled": false
    },
    "mcp": {
      "enabled": false
    }
  }
}
```

Setting `skills.enabled` to `false` disables the Skills Gateway management tools, skill fetching, and system-prompt injection. Setting `mcp.enabled` to `false` disables LiteLLM MCP discovery and tool registration. Restart Pi after changing these settings so previously registered tools are removed.

## Use

```
/model
```

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_API_KEY_HELPER` | unset | Command that prints a fresh LiteLLM bearer token. Takes precedence over `LITELLM_API_KEY`. The extension runs it while resolving request auth, and Pi's per-request auth path is uncached, so rotating/short-lived tokens stay fresh. |
| `LITELLM_HEADERS` | unset | JSON object of extra headers sent to LiteLLM provider, discovery, MCP, and Skills Gateway requests. Provider aliases can use it with `"headers": "$LITELLM_HEADERS"`. |
| `LITELLM_GCLOUD_TOKEN_AUTH` | unset | If set to a non-empty value other than `0`, use Google Application Default Credentials as the LiteLLM bearer token source. This takes precedence over `LITELLM_API_KEY_HELPER` and `LITELLM_API_KEY` when no stored `/login litellm` credential exists. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google default ADC path | Optional path to an ADC JSON file used by `LITELLM_GCLOUD_TOKEN_AUTH`. If unset, the extension checks the default gcloud ADC locations. |
| `LITELLM_OFFLINE` | unset | If `1`, disable all model and MCP discovery, including post-login discovery; use cached models only |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Background and explicit discovery fetch timeout in ms; `0` disables automatic discovery |
| `LITELLM_VERBOSE_DISCOVERY` | unset | If `1`, enable progress messages during model and MCP discovery (login, refresh, startup); discovery is silent by default |
| `LITELLM_MODELS_DEV` | enabled | Set to `0` to disable models.dev metadata enrichment, including its cache and network request; `/v1/models` still uses Pi catalog metadata and defaults |

`LITELLM_DISCOVERY_TIMEOUT_MS=0` disables automatic and explicit refresh model discovery. It does not replace the base URL or API key settings required to send requests when you are not using `/login litellm`.

Models.dev metadata is cached in `litellm-models-dev.json` under the Pi agent directory for 28 days. Fresh data avoids the public request; stale data is used immediately while one background refresh updates the cache. Set `LITELLM_MODELS_DEV=0` when your LiteLLM metadata is authoritative and no external enrichment is wanted.

### Google ADC token auth

When your LiteLLM proxy accepts Google OAuth access tokens, you can let the extension refresh tokens from Application Default Credentials:

```bash
gcloud auth application-default login
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_GCLOUD_TOKEN_AUTH=1
```

Only `authorized_user` ADC files are supported. Service account JSON files are rejected with a warning. Tokens are refreshed in-process when Pi resolves request auth and cached in memory for 50 minutes.

## LiteLLM MCP tools

If your LiteLLM proxy exposes MCP REST endpoints, this extension discovers tools from:

- `GET /mcp-rest/tools/list`
- `POST /mcp-rest/tools/call`

Each discovered tool is registered as a native Pi tool named `mcp_<server>_<tool>`, with simple JSON Schema parameters mapped to Pi/TypeBox parameters. Complex schemas fall back to a single `args` object. MCP discovery runs after Pi refreshes LiteLLM models or after `/login litellm`; extension activation never waits for it. MCP tools run in Pi's parallel tool mode and retry transient failures once.

## LiteLLM Skill Hub

If your LiteLLM proxy exposes `/claude-code/marketplace.json`, enabled skills are fetched before each agent turn and appended to the system prompt as a `litellm_skills` section. The extension falls back to the legacy `/v1/skills` Skills Gateway path when Skill Hub is unavailable. It also registers Pi tools for basic skill management:

- `litellm_skill_list`
- `litellm_skill_create`
- `litellm_skill_delete`

## Mocked LiteLLM smoke workflow

The `LiteLLM Smoke` GitHub Actions workflow starts VidaiMock and a real LiteLLM proxy on the runner. LiteLLM exposes OpenAI-compatible and Anthropic routes whose upstreams are served by VidaiMock, then this extension's smoke runner discovers those models through LiteLLM and sends `/v1/chat/completions` requests through the proxy.

This keeps the LiteLLM integration path under test but does not call real LLM APIs. No provider API keys or GitHub Models permission are required. The smoke runner also asserts that discovery came from `/model/info` (`LITELLM_SMOKE_EXPECT_SOURCE`) so a silent fallback to `/v1/models` fails the run. The workflow also runs auth checks plus optional Postgres-backed auth checks when `LITELLM_LICENSE` is configured for virtual-key and admin-route behavior, then runs a non-interactive Pi CLI smoke with `--list-models` and `-p` against both the OpenAI-compatible and Anthropic-backed routes, so extension loading, model discovery, and real completion paths are covered without opening the TUI. It also runs an interactive Pi TUI smoke covering `/login litellm` and Pi's native `/model` refresh.

## Development

This package requires Node.js `>=22.19.0`. CI currently uses Node `26.5.0`.

```bash
npm ci
npm run check
npm run clean && npm run build
```

`npm run check` runs Biome, type checking, the Vitest suite, and the supply-chain package-content guard. Pi installs and local smoke checks load the shipped `src/index.ts` entrypoint directly; `dist/` is verification output only.

Before changing package contents or dependency policy, also run:

```bash
npm run supply-chain:guard
npm pack --dry-run
```

The published npm package contains only `src`, `README.md`, `LICENSE`, and the `package.json` npm always includes. Pi loads the TypeScript source entrypoint for both npm and Git installs.

## Release

Releases are driven by semver tags named `v*.*.*`. The GitHub release workflow installs from the lockfile, runs the checks, builds `dist`, verifies the package tarball, publishes to npm with provenance, and creates a GitHub release.

Before tagging a release, keep `package.json` and `package-lock.json` versions in sync and verify the dry-run package contents.

## Model catalog

Dynamic catalogs are persisted by Pi in `~/.pi/agent/models-store.json`. Credentials remain in `~/.pi/agent/auth.json`. The legacy `litellm-models.json` cache is ignored and is not deleted; `litellm-models-dev.json` is the live models.dev cache described above and is still used.

Opening `/model` refreshes configured provider catalogs in the background using Pi's native model lifecycle.

### Model host enforcement

Models this provider dispatches are sent to the host the active credential resolves to. A model that cannot be matched to that host is hidden from `/model` rather than requested, and its request URL is derived from the credential's root rather than from whatever base URL the model carries.

Two separate mechanisms are involved, and they behave differently: **catalog filtering** decides what `/model` offers, and the **dispatch guard** decides whether a chosen model may issue a request. Choosing a model by id, or restoring one from a saved session, skips filtering entirely and goes straight to dispatch.

**Catalog filtering** is uniform. A model is offered only when its protocol is one discovery selects and its host matches the active credential; everything else is dropped with a diagnostic, per the table below.

**Dispatch** depends on how Pi routes the model, and Pi routes to a provider only when that provider currently lists a model using the same `api`:

| Model reaching dispatch | Behavior |
|---|---|
| `api` matching a model this provider currently lists, host matches | sent to the credential's host, URL derived from that host |
| `api` matching a model this provider currently lists, host differs | rejected before any request |
| `api` with **no** currently-listed model — including `openai-responses` on a proxy that exposes no responses routes | **not contained**; open upstream defect, see below |

The third row is wider than a foreign `api`: if your proxy exposes no `mode: "responses"` routes, then `openai-responses` has no listed model, and a hand-written `models.json` entry using it is dispatched by Pi through its global API registry without calling this provider. The request goes to whatever `baseUrl` the entry names, carrying this provider's API key and configured headers. Such a model is still kept out of `/model`, and it is rejected if it ever does reach this provider, but neither prevents a direct or session-restored dispatch.

That third row is an open defect, not intended behavior, and it is not something this provider can close: `Provider` exposes no way to declare which protocols an implementation supports, so Pi has only the current model list to route on. The fix belongs in Pi — route by a provider's declared protocols, or expose that declaration — and until it lands, do not point a `models.json` LiteLLM entry at a host you do not intend to receive your proxy credentials.

The behavior in all three rows is asserted against Pi's real composer in `tests/dispatch-routing.test.ts`, including the credential exposure in the third. That test is written to fail if Pi starts routing by declared protocol, which is the signal that the row can be corrected.

A model is dropped when:

| Reason | Diagnostic mentions | Fix |
|---|---|---|
| Nothing configured — no base URL resolves and no credential supplies one | `no LiteLLM base URL is configured` | Set `LITELLM_BASE_URL` or run `/login litellm` |
| Base URL is not a valid `http`/`https` URL, such as `localhost:4000` with no scheme | `invalid LiteLLM base URL` / `invalid LiteLLM model URL` | Correct `LITELLM_BASE_URL`; include the scheme |
| Base URL is still the documentation placeholder `https://litellm.example.com` | `placeholder LiteLLM model host` | Set your proxy's URL, or run `/login litellm` |
| A cached model's URL is unparseable | `Cached model has an invalid LiteLLM model URL` | Refresh the catalog by opening `/model` |
| A cached model's host is the placeholder | `Cached model uses a placeholder LiteLLM model host` | Refresh the catalog by opening `/model` |
| A cached model was discovered against another host, as after switching proxies | `stale LiteLLM model host` | Refresh against the current proxy by opening `/model` |
| The model declares an `api` this extension does not implement | `declares unsupported protocol` | Correct `api` in `models.json` (see [Protocols](#protocols)) |

Each distinct diagnostic is written once per session to stderr; the message names the model or host, so several offending models produce several lines. Opening `/model` fixes only the cached-model rows — the first three are configuration problems a refresh cannot resolve.

Because enforcement is keyed on the credential, `LITELLM_OFFLINE=1` does not recover a host mismatch on its own: offline mode serves the persisted catalog, and a catalog discovered against another host is still dropped. Clear the mismatch with one online refresh against the current proxy, after which offline mode works normally again.

### Protocols

Discovery maps each model to a request protocol and derives its request URL from the proxy root:

| `api` | Request base | Selected when |
|---|---|---|
| `openai-completions` | `<root>/v1` | default for chat-style routes |
| `openai-responses` | `<root>/v1` | `/model/info` reports `mode: "responses"` |

A model in `~/.pi/agent/models.json` whose `api` is anything other than these two is kept out of `/model` and rejected if it reaches this provider, with a diagnostic naming the supported values.

These two protocols are the whole set this provider implements. Anthropic Messages support is a separate change; a protocol is added here only together with the discovery mapping that selects it, because Pi will not route a protocol this provider never lists.

For models this provider dispatches, a per-model `baseUrl` in `models.json` is not honored as a request target — it is re-derived from the active credential host, and the model is rejected when its host differs.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| "discovered no models" | Proxy returned an empty list — check pi's startup log and verify `/model/info`, `/v1/models`, or `/health` responds |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models. Offline mode does not recover a host mismatch — see [Model host enforcement](#model-host-enforcement) |
| A provider shows no models at all | The base URL is missing, scheme-less, still the placeholder, or differs from the host the cached models were discovered against. Check stderr for the reason and match it against the table in [Model host enforcement](#model-host-enforcement) |
| A model configured in `models.json` never appears | Its `api` is not one of the supported protocols, or its `baseUrl` host differs from the active credential host — see [Protocols](#protocols) |
| `401 Token expired` | Set `LITELLM_API_KEY_HELPER`. |
| No models with gcloud auth | Verify `gcloud auth application-default login` has been run or set `GOOGLE_APPLICATION_CREDENTIALS` to an `authorized_user` ADC file |
| Enterprise SSO login shows "virtual key generation failed" | The LiteLLM instance may lack a database (`/key/generate` requires one), your user account may lack key-generation permission, or the request timed out; the JWT is used directly as a fallback |
| Enterprise SSO token prompt fails with "SSO token is required" | The token field was left empty — paste the token copied from the LiteLLM UI |
| MCP tools not showing | Verify the proxy exposes `/mcp-rest/tools/list` and open `/model` after fixing the proxy |
| Skills not affecting prompts | Verify the proxy exposes `/claude-code/marketplace.json` or `/v1/skills` and returns enabled skills |

## License

MIT — see [LICENSE](./LICENSE).
