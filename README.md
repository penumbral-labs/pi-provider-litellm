# pi-provider-litellm

LiteLLM proxy native Provider extension for [Pi](https://pi.dev). Pi 0.83.0+ is required.

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
build step and no `node_modules`. Install dependencies only to run the test suite or local checks.

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
| `baseUrl` | `LITELLM_BASE_URL` for `litellm`; required for aliases | LiteLLM proxy URL, with or without `/v1` |
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

## Model transport

A `/model/info` route group uses native Anthropic `/v1/messages` only when every deployment explicitly reports Chat
mode, has positive Claude family/model evidence through an Anthropic-, Bedrock-, or Vertex-capable adapter, and resolves
the same Anthropic compatibility policy. Provider-prefixed and valid unprefixed Claude model identifiers are supported;
adapter names alone never select Messages. Mixed, unknown, and fallback-only groups remain on Chat Completions;
unanimous explicit Responses mode takes precedence. The underlying catalog/provider identity and pricing are unchanged
by this choice.

Requiring unanimous compatibility matters when one public alias is load-balanced across two Claude generations, as
happens mid-migration: if one deployment requires adaptive thinking and another requires budget thinking, no single
Messages request satisfies both, so the group stays on Chat Completions and LiteLLM adapts the payload per deployment.
A backend the bundled catalog does not recognize is treated the same way, rather than guessing a request shape.

Native Messages authenticates to LiteLLM with `x-api-key` (not `Authorization: Bearer`) and intentionally omits
`litellm_session_id`; LiteLLM session grouping remains enabled for OpenAI Chat and Responses requests.

Anthropic request compatibility — adaptive versus budget thinking, whether the route accepts `temperature`, and strict
tool schemas — comes from the backend model LiteLLM reports, never from the public route name. Deployment decoration is
canonicalized first, so Bedrock cross-region and cross-partition inference profiles, Bedrock model versions, Vertex
serving suffixes, and dated release snapshots resolve to the same policy as the model they are a snapshot of. The same
canonical identifiers are looked up in the adapter's own catalog, so a decorated Bedrock Claude route keeps its Amazon
Bedrock provider identity, prices, limits, and thinking levels. A route whose backend has no entry in the bundled catalog
at all (Vertex Claude today) still carries compatibility while its identity and pricing stay withheld.

LiteLLM returns `x-litellm-response-cost` on `/v1/messages`, so native Messages routes use the same actual-cost reporting
path as Chat Completions and Responses. If an older proxy omits that header, Pi retains the discovered static estimate.

## Model catalog authority

Metadata resolves per field, from two sources with a fixed precedence.

**Explicit router values win.** Anything `/model/info` reports for a deployment — capabilities, context and output limits, per-token costs — is used as reported, whether or not the catalog knows the model. Non-finite or non-positive limits are ignored rather than clamping the group.

**Catalog values fill the gaps, but only with bounded unambiguous identity.** Enrichment from Pi's bundled catalog requires a declared adapter, a known provider prefix, or a recognized bare Anthropic alias, **and** an entry for that backend model in the catalog of that provider. A declared adapter is authoritative: a Vertex-served Claude is not priced from the first-party Anthropic catalog. An unqualified route such as a bare `gemini-2.5-pro` or `kimi-k2-thinking` deliberately stays unknown rather than adopting metadata from whichever provider catalog lists that name first, and a model newer than the bundled catalog stays unknown even when its provider is obvious.

**Display cost is marked, not invented.** A model's name carries a `(no metadata)` or `(incomplete metadata)` suffix whenever any of the four display-cost fields (input, output, cache read, cache write) is unresolved, so an unknown price is never shown as free. Filling in only input and output costs therefore leaves the marker in place. For a multi-deployment route, limits reduce to the group minimum and resolved costs to the group maximum.

Qualifying the route in LiteLLM (`model_info.base_model`, `litellm_params.model`, or a provider-prefixed `model_name`) supplies the identity half; reporting all four cost fields supplies the pricing half.

## When LiteLLM models disappear

Models are offered only while the host they were discovered against still matches the active credential's host. After changing `LITELLM_BASE_URL`, switching credentials, or logging in against a different proxy, open `/model` to refresh discovery. Until then LiteLLM contributes no models and the reason is written once to stderr. This is deliberate: a cached model is never silently re-pointed at a host it was not discovered from.

A Claude route discovered before this version was cached as a Chat Completions model. It keeps working on that transport and moves to native Messages on the next `/model` refresh.

## Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `LITELLM_API_KEY_HELPER` | unset | Command that prints a fresh LiteLLM bearer token. Takes precedence over `LITELLM_API_KEY`. Registered as a `!command` provider key; Pi re-runs it on every request (the per-request auth path is uncached), so rotating/short-lived tokens stay fresh. |
| `LITELLM_HEADERS` | unset | JSON object of extra headers sent to LiteLLM provider, discovery, MCP, and Skills Gateway requests. Provider aliases can use it with `"headers": "$LITELLM_HEADERS"`. |
| `LITELLM_GCLOUD_TOKEN_AUTH` | unset | If set to a non-empty value other than `0`, use Google Application Default Credentials as the LiteLLM bearer token source. This takes precedence over `LITELLM_API_KEY_HELPER` and `LITELLM_API_KEY` when no stored `/login litellm` credential exists. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google default ADC path | Optional path to an ADC JSON file used by `LITELLM_GCLOUD_TOKEN_AUTH`. If unset, the extension checks the default gcloud ADC locations. |
| `LITELLM_OFFLINE` | unset | If `1`, disable all model and MCP discovery, including post-login discovery; use cached models only |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Background and explicit discovery fetch timeout in ms; `0` disables automatic discovery |
| `LITELLM_VERBOSE_DISCOVERY` | unset | If `1`, enable progress messages during model and MCP discovery (login, refresh, startup); discovery is silent by default |
| `LITELLM_MODELS_DEV` | enabled | Set to `0` to disable models.dev metadata enrichment, including its cache and network request; `/v1/models` still uses Pi catalog metadata for provider-qualified ids and conservative defaults otherwise |

`LITELLM_DISCOVERY_TIMEOUT_MS=0` disables automatic and explicit refresh model discovery. It does not replace the base URL or API key settings required to send requests when you are not using `/login litellm`.

Models.dev metadata is cached in `litellm-models-dev.json` under the Pi agent directory for 28 days. Fresh data avoids the public request; stale data is used immediately while one background refresh updates the cache. Set `LITELLM_MODELS_DEV=0` when your LiteLLM metadata is authoritative and no external enrichment is wanted.

### Google ADC token auth

When your LiteLLM proxy accepts Google OAuth access tokens, you can let the extension refresh tokens from Application Default Credentials:

```bash
gcloud auth application-default login
export LITELLM_BASE_URL="https://litellm.your-domain.com"
export LITELLM_GCLOUD_TOKEN_AUTH=1
```

Only `authorized_user` ADC files are supported. Service account JSON files are rejected with a warning. Tokens are cached in memory for 50 minutes and the registered provider key is a Pi `!command`, so request-time auth resolves a fresh token when Pi sends model requests.

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

The `LiteLLM Smoke` GitHub Actions workflow starts VidaiMock and a real LiteLLM proxy on the runner. LiteLLM exposes
route-distinct Chat, Responses, native Messages, and mixed-deployment models whose upstreams are served by VidaiMock.
The smoke runner discovers those models through LiteLLM, asserts each model's expected API, requires the union of
`/v1/messages`, `/v1/chat/completions`, and `/v1/responses`, and checks the configured `x-litellm-response-cost`
expectation. The workflow also proves endpoint coverage from captured LiteLLM request logs rather than response text, and
requires at least two `/v1/messages` requests so the Pi CLI's own native Messages request is proven and not merely
the runner's. The smoke runner authenticates with `Authorization: Bearer`, so it covers route reachability rather than
the `x-api-key` path the extension uses; that path is covered by the Pi CLI smoke and by wire-compatibility tests.

This keeps the LiteLLM integration path under test but does not call real LLM APIs. No provider API keys or GitHub Models permission are required. The smoke runner also asserts that discovery came from `/model/info` (`LITELLM_SMOKE_EXPECT_SOURCE`) so a silent fallback to `/v1/models` fails the run. The workflow also runs auth checks plus optional Postgres-backed auth checks when `LITELLM_LICENSE` is configured for virtual-key and admin-route behavior, then runs a non-interactive Pi CLI smoke with `--list-models` and `-p` against both the OpenAI-compatible and Anthropic-backed routes, so extension loading, model discovery, and real completion paths are covered without opening the TUI. It also runs an interactive Pi TUI smoke covering `/login litellm` and Pi's native `/model` refresh. One route is configured as two deployments sharing a `model_name`, so the workflow proves LiteLLM really returns a row per deployment carrying `model_info.id`, `mode`, `supported_openai_params`, and `allowed_openai_params` — the upstream contract deployment-group reduction depends on.

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

Dynamic catalogs are persisted by Pi in `~/.pi/agent/models-store.json`. Credentials remain in `~/.pi/agent/auth.json`. Legacy `litellm-models*.json` files are ignored and are not deleted.

Opening `/model` refreshes configured provider catalogs in the background using Pi's native model lifecycle.

### Deployment groups and metadata evidence

`/model/info` returns one row per deployment, so several rows can share a `model_name`. Those rows are reduced to a single model using the least capable value from each deployment: the smallest context window and max tokens, the highest price, vision and reasoning only when every deployment reports them, and Pi catalog metadata only when the deployments agree on which backend serves the route.

Reasoning controls come from deployment evidence rather than the route name, and no thinking level is ever offered that the model's compatibility evidence cannot actually put on the wire. Chat and Responses discovery share that rule: an explicit vendor `supportsReasoningEffort: false` conclusion leaves both APIs with no selectable level, even though the Chat-only flag itself is not copied into Responses compatibility metadata.

For backends whose reasoning contract is recognized — the Kimi K2.5/K2.6, K2.7 Code and K3 generations, and DeepSeek V4 — a level is offered only when a deployment parameter can carry it. The carrying parameter is generation-specific, and **every** deployment in the group must declare it, because the group reduces to the parameters they all accept:

| Recognized backend | Carrying parameter |
|---|---|
| Kimi K2.5 / K2.6 | `thinking` |
| Kimi K2.7 Code | `thinking` (thinking cannot be switched off) |
| Kimi K3 | `reasoning_effort` |
| DeepSeek V4 | `thinking`, `reasoning_effort`, or both |

Without that evidence the route is reported as reasoning-capable with no selectable level, rather than offering levels that would be silently dropped. Other Kimi generations have no recognized contract, so they fail closed the same way; a carrier is never guessed from the route name or from vendor documentation. Backends outside these families keep whatever the Pi catalog describes and use the standard `reasoning_effort` field.

Declaring the parameter your backend accepts, on every deployment of the route, is the config-side fix.

Two suffixes can appear in a displayed model name. Which one you see depends on the discovery source, not on how many deployments a route has:

| Suffix | Meaning |
|---|---|
| `(incomplete metadata)` | A reduced `/model/info` group without complete price evidence, at any deployment count. Kept as the conservative floor and never enriched, because a deployment's metadata cannot be re-derived from the route name. LiteLLM omits the cache-price fields for backends that have no cache pricing, so this appears on ordinary routes that are otherwise fully described. |
| `(no metadata)` | An evidence-free `/v1/models` model, where deployments are not exposed at all. Enriched from the Pi catalog on a later read if the id becomes resolvable — and only while its cached compatibility settings can actually carry what the catalog offers. |

Catalog enrichment needs a provider it can trust. On `/model/info` those signals are the deployment's `litellm_params.model` and `model_info.base_model`, resolved through `model_info.litellm_provider`; on the `/v1/models` fallback it is `owned_by` or a provider-qualified id such as `openai/gpt-5.5`. Recognized Anthropic aliases (`sonnet-4-6`, `opus-5`, dated snapshots) are also resolved, because they canonicalize onto a single Anthropic catalog id.

A route name never overrides deployment evidence. It is used as a fallback in two bounded cases: when a route reduces to exactly one deployment and nothing else identified the backend, and when no deployment identifies a backend family at all — in which case a Kimi- or Claude-shaped route name can select catalog metadata, reasoning controls, token-field compatibility, or response-only `<think>` normalization. A route name never authorizes Moonshot strict tool-message repair, because that changes outbound assistant messages. The repair requires every deployment in the route group to identify the Moonshot/Kimi family. A route named only `mistral-large-latest` matches no catalog id and stays on conservative defaults.

When deployments disagree about which backend serves a route, or when only some of them identify one, catalog limits, pricing, and reasoning metadata are withheld for the whole group. Disagreements where at least one deployment resolved a provider are additionally reported on stderr with a list bounded to three affected route names. If only some deployments identify Moonshot/Kimi, strict tool-message repair is withheld and a warning names at most three affected routes. Both diagnostics are deduplicated once per process per route: refreshes do not repeat a persistent warning, but a newly affected route is still reported. Cached models created before this policy existed also keep that outbound repair off until a successful refresh records unanimous deployment evidence.

Discovery falls back from `/model/info` to `/v1/models` to `/health`, and capability shrinks along that path because the available evidence does. Only `/model/info` exposes complete deployment groups; `/health` reports one deployment at a time and is not reduced across a route's deployments, so a multi-deployment route discovered that way reflects whichever deployment `/health` listed first. Prefer a key that can read `/model/info` when reasoning controls matter.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| "no credentials" warning at startup | Env vars not set and no OAuth credential — run `/login litellm` |
| "discovered no models" | Proxy returned an empty list — check pi's startup log and verify `/model/info`, `/v1/models`, or `/health` responds |
| No LiteLLM models, with a "stale LiteLLM model host" note on stderr | Cached models were discovered against a different host than the active credential — open `/model` to refresh |
| No LiteLLM models, with a "placeholder LiteLLM model host" note | `LITELLM_BASE_URL` or a stored credential is literally the sample `litellm.example.com`, which is never contacted — set your own proxy URL |
| No LiteLLM models and no note at all | No base URL is configured anywhere — set `LITELLM_BASE_URL` or run `/login litellm` |
| No LiteLLM models, with an "invalid LiteLLM model URL" note | `LITELLM_BASE_URL` is not an absolute `http(s)` URL (for example `localhost:4000` instead of `http://localhost:4000`) |
| Model name ends in `(no metadata)` or `(incomplete metadata)` | The route reports incomplete cost data and its backend is not unambiguously identifiable — see [Model catalog authority](#model-catalog-authority) |
| `/model/info` returning 401/403/404 | Expected behavior with virtual keys — extension falls back to `/v1/models` |
| Discovery times out | Increase `LITELLM_DISCOVERY_TIMEOUT_MS` or set `LITELLM_OFFLINE=1` to fall back on cached models |
| Model shows as reasoning-capable but no thinking level can be selected | The route has no usable reasoning-control evidence. For a recognized backend, add its carrying parameter (see [the table above](#deployment-groups-and-metadata-evidence)) to `supported_openai_params` or `allowed_openai_params` on **every** deployment of the route. Other backends have no recognized contract and stay closed by design |
| Warning says strict tool-message repair was withheld | At least one route deployment identifies Moonshot/Kimi but another does not. Declare a Moonshot/Kimi backend model on **every** deployment in the group; renaming the public route is not enough. Until then, Moonshot tool calls on that route may fail |
| Model name ends in `(no metadata)` or `(incomplete metadata)` | Deployment pricing is incomplete and no catalog entry matched — see [Deployment groups and metadata evidence](#deployment-groups-and-metadata-evidence) |
| `401 Token expired` | Set `LITELLM_API_KEY_HELPER`. |
| No models with gcloud auth | Verify `gcloud auth application-default login` has been run or set `GOOGLE_APPLICATION_CREDENTIALS` to an `authorized_user` ADC file |
| Enterprise SSO login shows "virtual key generation failed" | The LiteLLM instance may lack a database (`/key/generate` requires one), your user account may lack key-generation permission, or the request timed out; the JWT is used directly as a fallback |
| Enterprise SSO token prompt fails with "SSO token is required" | The token field was left empty — paste the token copied from the LiteLLM UI |
| MCP tools not showing | Verify the proxy exposes `/mcp-rest/tools/list` and open `/model` after fixing the proxy |
| Skills not affecting prompts | Verify the proxy exposes `/claude-code/marketplace.json` or `/v1/skills` and returns enabled skills |

## License

MIT — see [LICENSE](./LICENSE).
