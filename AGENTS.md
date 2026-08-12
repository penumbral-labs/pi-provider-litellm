# Agent Notes

## Project Shape

- This package is a Pi extension that registers a `litellm` provider from `src/index.ts`.
- Source is TypeScript ESM under `src/`; tests are Vitest specs under `tests/`.
- Build output is `dist/`; do not edit generated output by hand.
- Adding a `src/` module means adding it to the `allowedPackageFiles` allowlist in `scripts/supply-chain-guard.ts` in the same commit, or `npm run prepublishOnly` fails.
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

- Model discovery lives in `src/discover.ts`.
- Prefer `/model/info` for rich metadata; fallback to `/v1/models` only on 401, 403, or 404.
- The `/v1/models` fallback enriches metadata from the Pi catalog and `https://models.dev/api.json`; keep fallback metadata tests current.
- Keep `LITELLM_OFFLINE` and `LITELLM_DISCOVERY_TIMEOUT_MS` behavior compatible with README docs.
- Stored Pi `/login litellm` credentials take precedence over `LITELLM_API_KEY`.
- Model catalogs are persisted by Pi in `~/.pi/agent/models-store.json`; this extension owns no model cache. `src/cache.ts` provides only `writeJsonAtomic`, used for the models.dev catalog it writes as `litellm-models-dev.json`. The legacy `litellm-models.json` cache is ignored and never deleted.

## LiteLLM Request Hooks

- `before_provider_request` is a global Pi hook. Only mutate provider payloads when `ctx.model?.provider === "litellm"`.
- Do not add user-facing flags or environment variables to hide provider-scoping bugs.
- `litellm_session_id` is optional LiteLLM session grouping metadata. If a LiteLLM server rejects it for LiteLLM-routed requests, keep Pi requests working first and document the admin-facing recommendation separately.
- Kimi/Moonshot responses may include `<think>` text; Pi-visible normalization happens in the `message_end` hook and should stay covered by feature tests.

## Protocols And Model Hosts

- `src/protocols.ts` is the single place a protocol's request base URL shape is defined. Add a `LiteLLMApi` member there and in `createLiteLLMProtocolApis()`; its explicit `Record` return type makes a missing entry a typecheck failure. Only add a protocol together with the discovery mapping that emits it: Pi routes to a provider only when that provider already lists a model using the same api, so a protocol this provider never produces is a protocol whose requests it cannot guard. `src/index.ts` still hardcodes `/v1` when composing the provider default and the discovery result; both survive only because `normalizeBaseUrl` strips one trailing `/v1` before `modelBaseUrl` runs.
- `api` values arriving from user `models.json` are unconstrained. Narrow with `isLiteLLMApi` before calling `resolveModelBaseUrl`. `toNativeModels` is exempt because its input is discovery output, which is typed.
- Host enforcement covers listing and dispatch separately. `filterModels` runs for every model Pi composes, including `models.json` entries, so the catalog is filtered either way. Dispatch is the gap: Pi's composer delegates to us only when we already list a model with the same `api`, so an entry whose `api` has no discovered models is dispatched through Pi's global API registry without reaching our guard. Do not describe the guarantee as unconditional, and do not describe `filterModels` as bypassed — only dispatch is.
- Pair `api` and `compat` by spreading one `modelProtocol()` result rather than assigning the two fields separately. What this buys is proven at runtime, by the `modelProtocol` unit tests and the discovery mapping tests — not at compile time in general: excess-property checking only rejects a *fresh literal* carrying a field unique to another protocol, and the two OpenAI compat types share enough optional members to be mutually assignable, so a mismatch assembled from typed values or a widened variable typechecks. The `@ts-expect-error` assertions cover the fresh-literal direction only.
- Any host remembered across calls must be bound to the credential it was resolved for, rank below an explicit per-request base URL, and be discarded when a non-OAuth credential resolves. All three are required: token binding stops a different credential reading it, ordering stops a stale value outranking the live credential, and discarding stops it surviving `/logout`.
- `getRuntimeAuth` runs from `before_agent_start` on every turn, including turns that never touch LiteLLM. It must return `undefined` for configuration problems rather than throw; only the LiteLLM tool surfaces should raise.
- `filterModels` must not throw: pi's `getAvailable()` has no per-provider isolation, so one throw empties every provider's model list. Reject a model by returning a reason and reporting it, never by throwing.
- The fail-closed host invariant is split deliberately: `src/index.ts` resolves and validates the credential root, `src/provider.ts` compares a model against that root. Both use the shared `isPlaceholderHost`; keep placeholder and scheme rules in those helpers instead of re-deriving them.
- Keep README's "Model host enforcement" table in sync when changing what hides a model, what a diagnostic says, or which fix resolves it.
- A guard is only pinned if deleting its body fails a test. When adding or changing one, delete the body and confirm a failure; if none appears, the guard is unpinned no matter how many tests pass.

## Compatibility Rules

- Provider-specific request compatibility belongs in discovered model `compat` metadata, not broad runtime mutation.
- Kimi/Moonshot-style models are handled in `completionsCompat()` and `responsesCompat()`, paired with their protocol by `modelProtocol()`; keep regression tests with model discovery changes.
- `compat` is per-protocol. `supportsStore`, `maxTokensField`, and `cacheControlFormat` are OpenAI-completions-only; `supportsDeveloperRole` also applies to `openai-responses`, where it defaults to true and must be disabled for Moonshot routes. Return `undefined` when a protocol needs no compat rather than an empty object.
- Anthropic-backed aliases need `cacheControlFormat: "anthropic"` so Pi forwards prompt-cache markers through LiteLLM on the completions path.

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
