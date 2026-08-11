# Agent Notes

## Project Shape

- This package is a Pi extension that registers a `litellm` provider from `src/index.ts`.
- Source is TypeScript ESM under `src/`; tests are Vitest specs under `tests/`.
- Build output is `dist/`; do not edit generated output by hand.
- The package entrypoint is `./dist/index.js`, and the Pi extension registration comes from `package.json` `pi.extensions`.
- Node support starts at `>=22.19.0`; GitHub workflows currently run Node `26.5.0`.

## Commands

- Use `npm ci` when reinstalling dependencies from the lockfile.
- Use `npm test` for the full test suite.
- Use `npm test -- tests/<file>.test.ts` for a focused Vitest run.
- Use `npm run check` before committing code changes; it runs Biome, typecheck, and tests.
- Use `npm run clean && npm run build` when changing exported/runtime code.
- Use `npm run supply-chain:guard` and `npm pack --dry-run` when package contents, dependency policy, or release packaging change.

## Discovery And Credentials

- Model discovery lives in `src/discover.ts`; deployment-group reduction lives in `src/model-groups.ts` and is the authority for reasoning capability.
- `/model/info` returns one row per deployment. Rows sharing a `model_name` are reduced conservatively (min limits, max price, unanimous capability). A route name never overrides deployment evidence; it is a last-resort fallback used only when a group reduces to exactly one deployment and nothing else identified the backend, plus the Anthropic alias canonicalization in `catalogLookupIds`.
- Reasoning controls for a recognized backend must come from `supported_openai_params` / `allowed_openai_params`, never from the route name and never from vendor documentation. The carrying parameter is generation-specific (K2.5/K2.6 and K2.7 Code use `thinking`, K3 uses `reasoning_effort`, DeepSeek V4 takes either) and must be accepted by every deployment in the group, since the reduction intersects accepted params. Live Azure deployments expose no reasoning controls for K2.6/K2.7 and only explicit `reasoning_effort` for K3/DeepSeek, so fail-closed is the default, not a fallback.
- Never advertise a `thinkingLevelMap` level the model's `compat` cannot transmit. A level needs either `supportsReasoningEffort` left enabled or an explicit `thinkingFormat`. Omitting the map is not fail-closed: pi-ai reads an absent map as every standard level supported, so deny each level explicitly. `advertisableLevels` in `src/model-groups.ts` is the single chokepoint where a level map meets the compat that carries it — every discovery path publishes through it, and both `tests/discover.test.ts` and `tests/provider-compat/stream.test.ts` sweep families and evidence shapes against it.
- `/health` reports one deployment at a time and is not reduced across a route's deployments, so a multi-deployment route discovered there reflects whichever deployment `/health` listed first (list order, not response order — `Promise.all` preserves list order and deduplication keeps the first entry). Fixing that needs group aggregation across the per-deployment detail fetches in `discoverFromHealth`; the approved PRD names it a non-goal, so it is deliberately not folded into unrelated changes.
- Contradictory family evidence — deployments naming different families, a routing model contradicting `base_model`, or only some deployments identifying a backend — is reported as `"conflicting"` rather than absent so it cannot decay into route-name inference.
- Do not add or change vendor replay/deferred-tool flags without primary evidence. Divergence from pi-ai's curated catalog is recorded as a residual, not silently reconciled.
- `litellmPolicy` carries both the request conclusion (`normalizeStrictToolMessages`) and the display conclusion (`normalizeThinkTags`). The `message_end` hook resolves the model through `ctx.modelRegistry` and reads that field; it must not re-derive a backend from the message's model id.
- Prefer `/model/info` for rich metadata; fallback to `/v1/models` only on 401, 403, or 404.
- The `/v1/models` fallback enriches metadata from the Pi catalog and `https://models.dev/api.json`; keep fallback metadata tests current.
- Keep `LITELLM_OFFLINE` and `LITELLM_DISCOVERY_TIMEOUT_MS` behavior compatible with README docs.
- Stored Pi `/login litellm` credentials take precedence over `LITELLM_API_KEY`.
- Cache data is stored as `litellm-models.json` under the Pi agent dir with a keyed API-key fingerprint and a 24-hour stale refresh window.

## LiteLLM Request Hooks

- `before_provider_request` is a global Pi hook. Only mutate provider payloads when `ctx.model?.provider === "litellm"`.
- Do not add user-facing flags or environment variables to hide provider-scoping bugs.
- Model-scoped request behavior travels on the discovered model as `litellmPolicy` (see `src/types.ts`), not as an id regex in the hook. `enrichCachedModel` re-derives it for cache entries that predate the policy, from the compatibility evidence already stored on the model — never from the route name, which is not evidence of a backend.
- `litellm_session_id` is optional LiteLLM session grouping metadata. If a LiteLLM server rejects it for LiteLLM-routed requests, keep Pi requests working first and document the admin-facing recommendation separately.
- Kimi/Moonshot responses may include `<think>` text; Pi-visible normalization happens in the `message_end` hook and should stay covered by feature tests.

## Compatibility Rules

- Provider-specific request compatibility belongs in discovered model `compat` metadata, not broad runtime mutation.
- Kimi/Moonshot-style models are handled in `buildCompat()`; keep regression tests with model discovery changes.
- Anthropic-backed aliases need `cacheControlFormat: "anthropic"` so Pi forwards prompt-cache markers through LiteLLM.

## Smoke And CI

- CI runs `npm ci` and `npm run prepublishOnly`.
- `.github/workflows/litellm-smoke.yml` uses VidaiMock plus a real LiteLLM proxy; it should not require real provider API keys.
- Keep smoke readiness probes bounded with `curl --connect-timeout 1 --max-time 3`.
- `scripts/smoke-runner.ts` exercises discovery and `/v1/chat/completions` through the proxy.
- The non-interactive Pi CLI smoke uses `./dist/index.js`, so runtime changes need a fresh build before running it.

## Release And Packaging

- The release workflow is tag-driven for `v*.*.*`; it publishes with `npm publish --access public --provenance` and creates a GitHub release.
- Local release prep should keep `package.json` and `package-lock.json` versions in sync, build `dist/`, run package checks, and create only local commits/tags unless the user explicitly overrides the no-push rule.
- Verify released state with `gh release view <tag>` and `npm view pi-provider-litellm version dist-tags --json` after the user pushes the tag.
- The npm package should stay limited to `dist`, `README.md`, and `LICENSE`.
- `scripts/supply-chain-guard.ts` rejects install lifecycle scripts, runtime dependencies, non-registry specs, non-registry lockfile URLs, and unexpected package files; update tests before changing that policy.

## Package Metadata

- Keep the Pi gallery image URL in `package.json` exactly as declared unless the user asks to change it.
- Do not include gallery assets in the npm package unless explicitly requested; verify packaging with `npm pack --dry-run` when package contents change.
