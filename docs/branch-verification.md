# Per-commit verification — `pr/multi-protocol-core`

Every commit on this branch was checked in an isolated `git archive` export with `node_modules` symlinked from the
worktree, so each row reflects that commit's tree alone rather than the branch tip.

Commands per commit: `npx tsgo --noEmit`, `npx biome check --error-on-warnings .`, `npx vitest --run`. The packaging
column is `npx tsgo -p tsconfig.build.json` followed by `npx tsx scripts/supply-chain-guard.ts`, which is what
`npm run prepublishOnly` runs and what CI enforces.

| # | Commit | Subject | Typecheck | Lint | Tests | Package guard |
|---|---|---|---|---|---|---|
| 1 | `5e57aeb` | feat: add multi-protocol provider core | pass | pass | 249 passed | **FAIL** |
| 2 | `d52db55` | fix: enforce LiteLLM credential hosts on request paths | pass | pass | **1 failed**, 254 passed | **FAIL** |
| 3 | `fffc9da` | fix: preserve OAuth host authority on request paths | **FAIL** | **FAIL** | not runnable | **FAIL** |
| 4 | `aa905e3` | style: normalize protocol reconstruction imports | pass | pass | **1 failed**, 256 passed | **FAIL** |
| 5 | `1c942aa` | test: remove MCP-only protocol fixture | pass | pass | 256 passed | **FAIL** |
| 6 | `d96a32a` | test: allow packaged protocol artifacts | pass | pass | 256 passed | pass |
| 7 | `4247a49` | fix: restore command-backed ADC availability | pass | pass | 256 passed | pass |
| 8 | `18e35e5` | fix: drop LiteLLM models with unknown protocols… | pass | pass | 261 passed | pass |
| 9 | `412fe86` | fix: scope the remembered LiteLLM OAuth host… | pass | pass | 262 passed | pass |
| 10 | `cc5254b` | fix: keep developer-role suppression for Moonshot Responses models | pass | pass | 265 passed | pass |
| 11 | `cf7670f` | test: assert custom headers reach MCP and Skills tool requests | pass | pass | 265 passed | pass |
| 12 | `6e5b5f0` | test: assert the offline LiteLLM restore performs no network calls | pass | pass | 265 passed | pass |
| 13 | `80180d7` | docs: document model host enforcement… | pass | pass | 265 passed | pass |

One test is skipped in every runnable row; that is pre-existing and unrelated.

## Red commits

### `fffc9da` does not build

```
src/index.ts(26,44): error TS2305: Module '"./mcp-tools.js"' has no exported member
'reportMcpRegistrationFailure'
```

`src/mcp-tools.ts` exports only `discoverMcpTools`, `executeMcpTool`, and `createMcpToolDefinitions` — at this commit and
at every other commit on the branch. The symbol never existed, so this is a half-written MCP-registration-isolation
feature that was committed and then abandoned. Biome fails on the same commit for the unused import plus a formatting
break at `src/index.ts:1035`.

### `d52db55` and `aa905e3` carry a failing test

`tests/index.test.ts` → "isolates MCP registration failures so valid sibling tools still register", added by `d52db55`,
asserts the string `"An MCP tool could not be registered."`, which appears nowhere in `src/`. It tests the same
never-implemented feature and fails from `d52db55` until `1c942aa` deletes it.

### Commits 1–5 fail the packaging guard

`5e57aeb` adds `src/protocols.ts`, but `scripts/supply-chain-guard.ts` only allowed the pre-existing `dist/*` modules
until `d96a32a`:

```
Supply-chain guard failed:
- npm package: unexpected published file dist/protocols.d.ts
- npm package: unexpected published file dist/protocols.js
```

So `npm run prepublishOnly` — the exact command CI runs — was red for five consecutive commits. The allowlist update
belonged in `5e57aeb`.

## Subjects that do not describe their change

| Commit | Subject | What it actually does |
|---|---|---|
| `aa905e3` | style: normalize protocol reconstruction imports | Removes the build-breaking `reportMcpRegistrationFailure` import and fixes an indentation break. Not a style change, and unrelated to protocols. |
| `1c942aa` | test: remove MCP-only protocol fixture | Deletes the failing MCP-registration-isolation test. Not a protocol fixture. |
| `d96a32a` | test: allow packaged protocol artifacts | Edits production packaging policy in `scripts/supply-chain-guard.ts`. Not a test change. |
| `4247a49` | fix: restore command-backed ADC availability | Reverts the gcloud `apiKeyConfig` regression that `fffc9da` introduced. Net diff versus the branch base is zero for both hunks, so the history contains a hidden revert of a regression that was never attributed to the commit that caused it. |

`git diff 54365fd..HEAD -- src/index.ts | rg gcloud` confirms the gcloud round trip: nothing survives to the tip beyond
import formatting.

## Recommendation

Commits 8–13 are green on every gate. Commits 1–7 leave three defects in the history: one commit that does not compile,
a test that fails for two commits, and a five-commit window where release packaging is broken.

None of this affects the branch tip, which passes all four gates. It matters for `git bisect`, for per-commit CI, and for
`AGENTS.md`'s requirement to run `npm run check` before committing.

The branch has no upstream:

```
$ git branch -vv
* pr/multi-protocol-core 80180d7 [no tracking entry]
```

so the history can be rewritten without a force-push to a remote. A rebase collapsing `fffc9da`/`aa905e3`/`1c942aa` into
`d52db55`, folding the allowlist change into `5e57aeb`, and dropping `4247a49` as a no-op would leave every commit green
with an honest subject.

**Not done here.** These commits are the frozen evidence the review was run against, so cleanup is deferred pending
explicit approval.
