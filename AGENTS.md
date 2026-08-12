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

- Model discovery lives in `src/discover.ts`; conservative deployment-group reduction lives in `src/model-groups.ts`.
- Prefer `/model/info` for rich metadata; fallback to `/v1/models` only on 401, 403, or 404.
- The `/v1/models` fallback enriches metadata from the Pi catalog and `https://models.dev/api.json` for ids that carry provider evidence — a known provider prefix, `owned_by`, or a bare Anthropic alias that canonicalizes to a `claude-` catalog id. Other unqualified ids stay unknown instead of being matched against every provider catalog. Keep fallback metadata tests current.
- Keep `LITELLM_OFFLINE` and `LITELLM_DISCOVERY_TIMEOUT_MS` behavior compatible with README docs.
- Stored Pi `/login litellm` credentials take precedence over `LITELLM_API_KEY`.
- Cache data is stored as `litellm-models.json` under the Pi agent dir with a keyed API-key fingerprint and a 24-hour stale refresh window.

## Deployment Groups And Metadata Authority

- `/model/info` returns one row per deployment. Rows sharing a `model_name` are reduced by `reduceModelGroup` before any transport, capability, limit, price, or compatibility decision. Never shallow-merge them.
- `reduceModelGroup` is pure and table-tested with an injected catalog resolver. Keep fetch, environment, clock, and provider scans out of it.
- A public route name is not backend evidence for a deployment group. It is still used in exactly two narrower places, both deliberate: as a catalog hint when the group reduces to exactly one routable deployment (the resolver's `singleton` argument), and as the input to `buildCompat`, which derives request-compatibility flags from the route id exactly as the baseline did. Do not widen either, and do not describe route text as wholly unused.
- Model-name suffixes are a behavioral contract, not decoration. ` (no metadata)` marks an evidence-free `/v1/models` model and authorizes `enrichCachedModel` to re-derive catalog metadata from the model id. ` (incomplete metadata)` marks a reduced `/model/info` group with incomplete price evidence, or an unresolved `/health` route, and is permanent: neither is ever re-authorized from the model id. A `/health` route the catalog resolves keeps its plain name. Adding a producer of either suffix is a contract change.
- Discovery and offline rehydration must produce identical models for `/model/info` groups and `/health` routes; `tests/provider.test.ts` covers those round trips. A hand-built cached model is not a round-trip test. `/v1/models` models are the intentional exception, since their sentinel permits bounded later enrichment.
- Three separate classifiers recognize Anthropic models, and they are intentionally distinct. `catalogLookupIds` canonicalizes ids and aliases and is the sole input to catalog-provider candidacy. `semanticFamily` recognizes the Claude family for reduction evidence. `ANTHROPIC_MODEL_PATTERN` drives `cacheControlFormat: "anthropic"` in `buildCompat`, without which prompt caching silently no-ops. Do not collapse them into one, and do not assume changing one is safe: any change to Anthropic recognition needs coordinated tests across all three.
- Withholding catalog authority is safe but invisible, so it emits a bounded stderr diagnostic that ignores `LITELLM_VERBOSE_DISCOVERY`. Keep such diagnostics to counts and public route ids.

## LiteLLM Request Hooks

- `before_provider_request` is a global Pi hook. Only mutate provider payloads when `ctx.model?.provider === "litellm"`.
- Do not add user-facing flags or environment variables to hide provider-scoping bugs.
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
