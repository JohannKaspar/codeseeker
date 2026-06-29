# codeseeker — fork orientation (`JohannKaspar/codeseeker`)

This is a **fork** of [`MotzWanted/codeseeker`](https://github.com/MotzWanted/codeseeker) — the released "Code Like Humans" (CLH) multi-agent ICD-coding pipeline. We use it to reproduce CLH end-to-end as the **baseline anchor** for the **agentic-icd** research project (tracked in the Obsidian research vault: experiment **E-007**, tasks **T-070 / T-071 / T-094**). It is **not** developed for upstream contribution — most of what we add is reproduction scaffolding.

> This file is the canonical orientation; `AGENTS.md` is a relative symlink to it (edit `CLAUDE.md`, never the symlink) so AGENTS.md-aware tools read the same guidance.

## Remotes

| remote | url | use |
| --- | --- | --- |
| `origin` | `JohannKaspar/codeseeker` (this fork) | push/pull our work |
| `upstream` | `MotzWanted/codeseeker` | pull upstream updates; **never push** |

Access: the **Mac** has SSH read/write to `origin` (authoring machine). **dobby** has it over HTTPS and is **pull-only** (its GitHub deploy key is read-only) — dobby changes reach the fork via the Mac, unless a write-capable key is added on dobby.

## Branches

- **`main`** — a **pristine mirror of `upstream/main`** (currently `d76d2c0`; upstream `main` has been dormant since 2026-01-22). **Never commit here.** Its only update path is `git fetch upstream && git merge --ff-only upstream/main && git push origin main`. Keeping it pristine is what makes `git diff main` answer "what have *we* changed" and lets us pull upstream (or cherry-pick a `feat/*` branch) cleanly if it ever revives.
- **`integration`** — **our working trunk.** All our feature branches merge here; both machines track it day-to-day, and this is where the reproduction infra below lives. Cut new work from here, merge it back here.
- **`feat/<task>`** — per-experiment/task feature branches, cut from `integration` and merged back into it. Name them after the vault task, e.g. `feat/T-095-…`.
- Upstream's own `feat/analyse-agent`, `feat/assign-agent`, `feat/mdace`, … branches are **MotzWanted's**, not ours — don't build on them directly; cherry-pick into `integration` if you need one.

### Workflow

```bash
git checkout -b feat/X integration          # start work
# … commit …
git checkout integration && git merge feat/X && git push origin integration
# absorb upstream (rare): update main ff-only from upstream, then:
git checkout integration && git merge main
```

## What's on `integration` (our additions, not upstream)

Reproduction infra for E-007 (CLH end-to-end on MDACE + the ACI-BENCH outpatient companion):

- **ACI-BENCH wiring** — `src/dataloader/aci/loader.py`, `src/dataloader/adapt/adapters/aci.py` (`AciBenchAdapter`), the `aci-bench-icd10cm` entry in `DATASET_CONFIGS`, and a `catalog_year` arg on `benchmark.Arguments` (FY2025 for ACI; the MDACE row-drop is guarded by dataset).
- **Runners** — `experiments/clh_run_{8b,70b,70b_s2,27b}.py` (MDACE, T-070/T-071) and `experiments/clh_aci_run_{8b,70b,27b}.py` (ACI, T-094). They drive `benchmark.run()` against a local vLLM (benchmark.py has no CLI — `Arguments` is a plain pydantic model). Run with `PYTHONPATH=src:experiments:_stubs`, `API_KEY=EMPTY`.
- **`_stubs/ml_datasets`** — stub for the proprietary `tanner` dependency so `dataloader/__init__` imports.
- **`src/dataloader/mdace/prepare_mdace.py`** — `trim_annotations` dict fix for polars 1.37.

Serving/env detail (dobby vLLM, the `vllm-q36` build for Qwen3.6-27B, Kerberos, local-disk practice) lives in the agentic-icd vault memory, not here. Scratch (`.qdrant_*/`, `*.bak`, experiment dumps) is git-ignored and never committed.
