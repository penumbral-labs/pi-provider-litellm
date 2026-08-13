# Agent Notes

## Project Shape

- This package is a Pi extension that registers a `litellm` provider from `src/index.ts`.
- Source is TypeScript ESM under `src/`; tests are Vitest specs under `tests/`.
- Build output is `dist/`; do not edit generated output by hand or publish it.
- Git and npm installs load `./src/index.ts` through `package.json` `pi.extensions`.
- Node support starts at `>=22.19.0`; GitHub workflows currently run Node `26.5.0`.

## Commands

- Use `npm ci` when reinstalling dependencies from the lockfile.
- Use `npm test` for the full test suite.
- Use `npm test -- tests/<file>.test.ts` for a focused Vitest run.
- Use `npm run check` before committing code changes; it runs Biome, typecheck, tests, and the package-content guard.
- Use `npm run clean && npm run build` when changing exported/runtime code.
- Use `npm run supply-chain:guard` and `npm pack --dry-run` when package contents, dependency policy, or release packaging change.

## Discovery And Credentials

- Model discovery lives in `src/discover.ts`; deployment-group reduction lives in `src/model-groups.ts` and is the authority for reasoning capability.
- `/model/info` returns one row per deployment. Rows sharing a `model_name` are reduced conservatively (min limits, max price, unanimous capability). A route name never overrides deployment evidence, but it is a bounded fallback in two places: a group that reduces to exactly one deployment may use it as a catalog hint, and a group where no deployment identifies a family still gets vendor request compat from a Kimi- or Claude-shaped name. The Anthropic alias canonicalization in `catalogLookupIds` is the single source for Anthropic recognition.
- The two name suffixes are chosen by discovery source, not deployment count: reduced `/model/info` groups use ` (incomplete metadata)` and are never enriched; only the evidence-free `/v1/models` path uses ` (no metadata)`, which authorizes catalog re-derivation on a later cache read.
- Reasoning controls for a recognized backend must come from `supported_openai_params` / `allowed_openai_params`, never from the route name and never from vendor documentation. The carrying parameter is generation-specific (K2.5/K2.6 and K2.7 Code use `thinking`, K3 uses `reasoning_effort`, DeepSeek V4 takes either) and must be accepted by every deployment in the group, since the reduction intersects accepted params. Live Azure deployments expose no reasoning controls for K2.6/K2.7 and only explicit `reasoning_effort` for K3/DeepSeek, so fail-closed is the default, not a fallback.
- Never advertise a `thinkingLevelMap` level the model's compatibility evidence cannot transmit. A Chat level needs either `supportsReasoningEffort` left enabled or an explicit `thinkingFormat`; a Responses model must also honor a vendor `supportsReasoningEffort: false` conclusion even though that Chat-only field is not copied into Responses compat. Omitting the map is not fail-closed: pi-ai reads an absent map as every standard level supported, so deny each level explicitly. `closeSerializerPolicy` in `src/model-groups.ts` is the single chokepoint where a level map meets the evidence for its serializer — every discovery path publishes through it, and both `tests/discover.test.ts` and `tests/provider-compat/stream.test.ts` sweep families and evidence shapes against it.
- `/health` reports one deployment at a time and is not reduced across a route's deployments, so a multi-deployment route discovered there reflects whichever deployment `/health` listed first (list order, not response order — `Promise.all` preserves list order and deduplication keeps the first entry). Fixing that needs group aggregation across the per-deployment detail fetches in `discoverFromHealth`; the approved PRD names it a non-goal, so it is deliberately not folded into unrelated changes.
- Withheld catalog-authority and strict-tool-repair diagnostics are both bounded to three public route ids and deduplicated once per process per route. A repeated refresh stays quiet, while a newly affected route still emits one line.
- Contradictory family evidence — deployments naming different families, a routing model contradicting `base_model`, or only some deployments identifying a backend — is reported as `"conflicting"` rather than absent so it cannot decay into route-name inference.
- Do not add or change vendor replay/deferred-tool flags without primary evidence. Divergence from pi-ai's curated catalog is recorded as a residual, not silently reconciled.
- `litellmPolicy` carries both the request conclusion (`normalizeStrictToolMessages`) and the display conclusion (`normalizeThinkTags`). The `message_end` hook resolves the model through `ctx.modelRegistry` and reads that field; it must not re-derive a backend from the message's model id. Strict tool-message repair requires unanimous discovered-family evidence across the route group; route-name evidence may influence reasoning effort, limits, token-field compatibility, or response-only normalization but never authorize that outbound rewrite. The forced-thinking sub-decision inside `moonshotPolicy` is still a route-id pattern, so pass it the route id and never a semantic label — no label can match that pattern, which would silently make the exemption unreachable and split the conclusion by discovery source.
- `closeSerializerPolicy` must gate every published level map, including the catalog enrichment inside `enrichCachedModel`, whose compat stays as stored.
- `buildCompat` takes no API argument. Per-API compat belongs to the multi-protocol core; returning an empty compat for Responses drops `supportsDeveloperRole` and makes the level map look transmissible when it is not.
- Prefer `/model/info` for rich metadata; fallback to `/v1/models` only on 401, 403, or 404.
- The `/v1/models` fallback enriches metadata from the Pi catalog and `https://models.dev/api.json` only when provider identity is unambiguous (known adapter, known provider prefix, or recognized Anthropic alias); keep fallback metadata tests current.
- Catalog authority is deliberately bounded: an unqualified id stays unknown rather than adopting metadata from an arbitrary provider catalog that lists the same name. Do not reintroduce an all-providers catalog scan.
- `http(s)`-only validation, canonical host identity, and placeholder rejection live in `src/host-policy.ts`; `normalizeBaseUrl` lives in `src/discover.ts`. Availability, request, and credential-resolution paths share these instead of re-implementing a check.
- Keep `LITELLM_OFFLINE` and `LITELLM_DISCOVERY_TIMEOUT_MS` behavior compatible with README docs.
- Stored Pi `/login litellm` credentials take precedence over `LITELLM_API_KEY`.
- Pi owns discovered-model persistence in `models-store.json`; this extension does not write a model cache. Legacy `litellm-models*.json` files are ignored and never deleted.
- The only agent-dir file this extension writes is `litellm-models-dev.json`, a models.dev metadata cache with a 28-day TTL.
- Google ADC is resolved in process through `src/gcloud-token.ts`; there is no helper subprocess and no `src/gcloud-token-cli.ts`. Only `authorized_user` credentials are supported, blank fields are rejected, and service accounts warn and fail closed.
- `apiKey.check` performs no network call, so it reports credential shape, not mintability. Its source label must mirror the precedence in `resolveCredentials`, so ADC is named whenever a complete ADC file exists; if the refresh token no longer mints, `resolve` falls back and reports the credential it actually used.

## LiteLLM Request Hooks

- `before_provider_request` is a global Pi hook. Only mutate provider payloads when `ctx.model?.provider === "litellm"`.
- Do not add user-facing flags or environment variables to hide provider-scoping bugs.
- Model-scoped request behavior travels on the discovered model as `litellmPolicy` (see `src/types.ts`), not as an id regex in the hook. For cache entries that predate the policy, `enrichCachedModel` may recover response-only normalization from the compatibility evidence already stored on the model, but it must not infer unanimous strict-tool-repair authority — and it never uses the route name as backend evidence.
- `litellm_session_id` is optional LiteLLM session grouping metadata. If a LiteLLM server rejects it for LiteLLM-routed requests, keep Pi requests working first and document the admin-facing recommendation separately.
- Kimi/Moonshot responses may include `<think>` text; Pi-visible normalization happens in the `message_end` hook and should stay covered by feature tests.

## Compatibility Rules

- Provider-specific request compatibility belongs in discovered model `compat` metadata, not broad runtime mutation.
- For native `anthropic-messages` routes, discovered `compat` is the only channel to pi-ai's serializer. Derive it from the backend model LiteLLM reports (never the public route name), canonicalize deployment decoration first, forward the carried fields as a unit, and require unanimity across the group. Unanimous positive compatibility is also a precondition for selecting Messages at all: a group that disagrees, or whose backend is unknown, must reduce to Chat Completions rather than route natively without it.
- Carrying backend compatibility must not grant catalog provider identity, pricing, or limits to a route that has not earned them.
- Kimi/Moonshot-style models are handled in `buildCompat()`; keep regression tests with model discovery changes.
- Anthropic-backed aliases need `cacheControlFormat: "anthropic"` so Pi forwards prompt-cache markers through LiteLLM.

## Smoke And CI

- CI runs `npm ci` and `npm run prepublishOnly`.
- The release workflow invokes `npm run prepublishOnly` explicitly before `npm publish`, so the publish gate still runs when npm lifecycle scripts are disabled.
- `.github/workflows/litellm-smoke.yml` uses VidaiMock plus a real LiteLLM proxy; it should not require real provider API keys.
- Keep smoke readiness probes bounded with `curl --connect-timeout 1 --max-time 3`.
- `scripts/smoke-runner.ts` exercises discovery and `/v1/chat/completions` through the proxy.
- The non-interactive Pi CLI smoke loads the package root (`-e .`) so it exercises `pi.extensions` resolution; the interactive terminal smoke loads `src/index.ts` by path.
- `--list-models` alone does not prove an extension loaded, because Pi also reports models from its own store. Assert a load-specific side effect instead.

## Release And Packaging

- The release workflow is tag-driven for `v*.*.*`; it publishes with `npm publish --access public --provenance` and creates a GitHub release.
- Local release prep should keep `package.json` and `package-lock.json` versions in sync, build `dist/`, run package checks, and create only local commits/tags unless the user explicitly overrides the no-push rule.
- Verify released state with `gh release view <tag>` and `npm view pi-provider-litellm version dist-tags --json` after the user pushes the tag.
- The npm package should stay limited to `src`, `README.md`, and `LICENSE`; builds are verification-only.
- `scripts/supply-chain-guard.ts` rejects install lifecycle scripts, runtime dependencies, non-registry specs, non-registry lockfile URLs, and unexpected package files, and requires every allowlisted source file to ship; update tests before changing that policy.

## Package Metadata

- Keep the Pi gallery image URL in `package.json` exactly as declared unless the user asks to change it.
- Do not include gallery assets in the npm package unless explicitly requested; verify packaging with `npm pack --dry-run` when package contents change.

## Protocols And Model Hosts

- `src/protocols.ts` owns protocol request-base projection. Keep protocol selection paired with its compat metadata.
- Validate LiteLLM request hosts and reject placeholder or stale cached hosts before sending credentials.
- Bind remembered OAuth hosts to the credential that established them; explicit request hosts rank first.
- Keep payload rewrites on a positive API allowlist so Chat/Responses rewrites cannot leak to Messages.
