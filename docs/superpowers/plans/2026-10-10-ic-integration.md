# IC Integration Plan — restore legacy initial conditions

**Goal:** Replace the null-IC stopgap with the legacy per-age initial-conditions
table, regenerated at runtime from the recovered 2019 recipe, **without changing
any learning/conversation logic**. A freshly constructed `BiLangModel` should again
have agents with realistic, graded linguistic knowledge at step 0.

**Branch:** `feature/update_all_project_dependencies_to_2026`. Commit after each task.

## Findings that fix the design (from git history, commit `34e5a87~1`)

- Legacy `set_lang_ics`: monolingual → own language from `'100_pct'`, other language
  null; bilingual → draw `k` from `ic_pct_keys = [10,25,50,75,90]`, `L1='{k}_pct'`,
  `L2='{100-k}_pct'`; `L12`/`L21` always null.
- ⇒ the table must contain exactly the six buckets **`{10, 25, 50, 75, 90, 100}`**,
  keyed `'{p}_pct'`. (`scripts/build_ics.py` currently has `{25,50,75,100}` → would
  `KeyError`.)
- Legacy `_set_lang_attrs(lang, pct_key)` copies row `age` of `t`, `S`, `wc`, derives
  `R`, `pct`, `mem_eff`, `excl_c`. Restore verbatim.
- `model.set_lang_ics_in_family` (parent→child bucket correlation via `biling_key`) is
  **unchanged since legacy**; restoring the two agent methods restores graded
  bilingualism with no new design.
- The `test_agent` `speak`-branch adaptation is a genuine correctness fix — keep it.
  `test_run_conversation` was skipped *because of* null ICs — attempt to revive.

## Design decisions

1. **Generator moves into the package**: `bilangsim/ic_generator.py`. The model must
   import it; `scripts/` is not part of the package. `scripts/build_ics.py` becomes a
   thin CLI (build/load + sanity report).
2. **Local RNG only.** The script's `np.random.seed(42)` would reset the *global* RNG
   mid-`__init__`, clobbering the model's seeds — and only on cache misses, making runs
   depend on cache state. The generator uses its own `np.random.RandomState(seed)`;
   sampling is the same `searchsorted(cdf, uniforms)` as `randZipf`.
3. **Reuse the model's CDFs.** `_build_ics()` runs right after `_build_cdfs()` and
   before `GeoMapper` (which creates agents), passing `self.cdf_data['s']`.
4. **Caching (Option A):** in-process memo (repeated constructions, e.g. tests) + disk
   cache keyed by `(vocab_red, max_steps, steps_per_year, buckets, seed,
   GENERATOR_VERSION)`. Location: `$BILANGSIM_CACHE_DIR`, else
   `$XDG_CACHE_HOME/bilangsim`, else `~/.cache/bilangsim`. On disk `t`/`S` as float32
   and `wc` as int32 (~130 MB); upcast to float64/int64 on load to match legacy dtypes.
   Arrays are marked read-only (agents `np.copy` them, as in legacy).
5. **Recipe fidelity:** the notebook recipe is kept as-is (incl. its single-then-bulk
   SM-2 application). The `2·act_c − 1` discrepancy with live `update_words_memory` is
   documented, not changed.

## Tasks

1. `bilangsim/ic_generator.py` + `tests/test_ic_generator.py` (keys/shapes/dtypes,
   determinism, global-RNG untouched, cache round-trip, read-only). Commit.
2. Rewrite `scripts/build_ics.py` as a thin CLI over the module. Commit.
3. Restore `_set_lang_attrs` + legacy `set_lang_ics` in `agent.py`; add
   `BiLangModel._build_ics()`; tests for mono/bilingual IC assignment. Commit.
4. Full suite vs. baseline (86 pass / 1 skip); try reviving `test_run_conversation`
   (`--count` for flakiness). Commit.
5. Docs: `CLAUDE.md` status, `docs/components/{model,agent}.md`, `run.py` warmup help
   text. Commit.

## Out of scope (follow-ups)

- Model snapshots (`pickle_model`) include `lang_ICs` (~260 MB RAM), as in legacy.
  Could be excluded via `BiLangModel.__getstate__` and re-loaded from cache on unpickle.
- The `2·act_c − 1` double application in `update_words_memory`.
- Calibration of the recipe (inflated `R@5y`), `spoken_only=False` (written CDFs).
