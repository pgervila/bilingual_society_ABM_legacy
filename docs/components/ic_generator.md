# `ic_generator.py` — linguistic initial-conditions table

`bilangsim/ic_generator.py` (~240 lines) builds the per-age table agents read their
starting linguistic knowledge from. It replaces the legacy
`lang_spoken_ics_vs_step.h5` file, which was never committed. `scripts/build_ics.py`
is a thin CLI over it.

## Provenance

The module re-implements the 2019 recipe from the deleted notebook
`lang_memory_model_research.ipynb` (recoverable with
`git show c44563d:lang_memory_model_research.ipynb`, cells 194/195/203):
`words_day_factor`, `get_lang_knowledge`, `study_memory_fun`. It is extended to
snapshot the state at every age, as legacy commit `18cad3c` did. Only the
spoken-only path is reproduced (`read_hours_per_day = 0`), which is what the legacy
model loaded.

## Table layout

```
lang_ICs['{p}_pct'][field][age, word]     p     in {10, 25, 50, 75, 90, 100}
                                          field in {'t', 'S', 'wc'}
                                          shape (max_life_steps, vocab_red)
```

`t` and `S` are float64, `wc` is int64. All arrays are **read-only**; agents
`np.copy` rows in `_set_lang_attrs`. `R`, `pct`, `mem_eff` and `excl_c` are not
stored, because agents derive them. The buckets are the ones legacy `set_lang_ics`
needs: `'100_pct'` for monolinguals, `'{k}_pct'` and `'{100-k}_pct'` for
bilinguals with `k in ic_pct_keys = [10, 25, 50, 75, 90]`.

## The recipe: `build_bucket_trajectory(pct_hours, cdfs, rng)` (`:69`)

A single solitary speaker, exposed from birth to a constant fraction `pct_hours` of
full daily token flow. Initial state: `S = 0.01`, `t = 100`, `wc = 0`, `R = 0`. Each
step (≈10 days):

1. **Token volume:** `n = int(F_S · pct_hours · (n_red / words_day_factor(age)) · 10)`,
   with `F_S = 2` (own speech, TV, radio). `words_day_factor` (`:57`) is the notebook's
   3-piece curve: `2.5 + 100·e^(−0.014·age)` before 14 y, `2.5` from 14 to 65 y,
   `1.5 + e^(0.002·(age−65y))` after. A larger factor means fewer tokens, so
   infants get few.
2. **Sampling:** `n` word indices from the age's CDF (`searchsorted(cdfs[age], uniforms)`,
   same as `randZipf`); `wc` += counts.
3. **Encoding gate:** only words with lifetime `wc > 5` are updated.
4. **SM-2 update** (`a, b, c, d = 7.6, 0.023, −0.031, −0.2`, `ΔS = a·S^(−b)·e^(c·100·R) + d`):
   one application, then `t` reset to 0 for activated words (+1 for the rest), then
   `act_c − 1` further activations in bulk on the updated `S`. When nothing passes
   the gate, all `t` += 1.
5. **Aging:** after 65 y, `S −= 0.01` per step (floor `1e−6`).
6. `R = exp(−k·t/S)`, `k = ln(10/9)`; snapshot `t`, `S`, `wc` into row `age`.

`build_lang_ics(cdfs, buckets, seed)` (`:130`) runs one trajectory per bucket with a
single private `np.random.RandomState(seed)`.

> **Known difference from live learning:** live `update_words_memory` applies
> `act_c` then `act_c − 1` activations (≈ `2·act_c − 1`), so agents learn faster
> after construction than the recipe that built their ICs. The recipe is kept as-is.
> See [agent.md](agent.md).

## RNG isolation

The generator never touches the global `random` or `numpy` RNG state. That matters
because it runs inside `BiLangModel.__init__`. Reseeding the global RNG there would
override the model's seeds, and only on cache misses, so runs would depend on cache
state. `tests/test_ic_generator.py::test_global_rng_untouched` guards this.

## Caching: `load_or_build_lang_ics(cdfs, buckets, seed, cache_dir=None, use_disk_cache=True)` (`:201`)

1. **In-process memo** (`_MEMO`): repeated model constructions in one process (e.g.
   the test suite) reuse the same table object.
2. **On-disk cache**: `lang_ics_<key>.npz`, where `key` hashes the CDF array
   content, the buckets, the seed and `GENERATOR_VERSION` (`:32`). Bump
   `GENERATOR_VERSION` whenever the recipe changes. The directory is
   `$BILANGSIM_CACHE_DIR`, else `$XDG_CACHE_HOME/bilangsim`, else `~/.cache/bilangsim`
   (`default_cache_dir`, `:147`). Writes are atomic (temp file + `os.replace`). A
   corrupt file is rebuilt, and an unwritable directory is skipped silently.
3. **Storage precision:** float32 `t`/`S` and int32 `wc` on disk (~130 MB for
   `vocab_red=500`), upcast on load. A cache miss returns the same quantized values
   a later hit would, so results never depend on whether the cache existed.

Costs (`vocab_red=500`, 3600 steps, 6 buckets): build ≈ 2.6 s, disk load ≈ 0.2 s,
~260 MB in RAM per model.

## CLI: `scripts/build_ics.py`

```bash
PYTHONHASHSEED=0 uv run python scripts/build_ics.py [--vocab-red 500] [--seed 42] [--no-cache]
```

It builds the CDFs with `zipf_generator.age_indexed_cdfs`, exactly as the model
does, so it warms the cache entry the model will read. It then prints sanity
statistics: `S` head/tail and encoded-word counts at 5 y and 30 y (100 % bucket),
and per bucket the cumulative tokens at 30 y, mean `R` and `pct(R > 0.9)` at
5/30/70 y.

## Calibration notes (not changed; for the model-improvement plan)

- Young partial-exposure speakers come out generous: 25 % exposure gives ≈37 % of
  words with `R > 0.9` at 5 y. The main drivers are the low encoding gate
  (`wc > 5` in compressed units), the large first-hit `ΔS` (≈8), the bulk update,
  `F_S = 2`, and the 10-days-per-step compression.
- With sustained exposure, old-age decay (`S −= 0.01`) is negligible against
  stabilities in the tens to tens-of-thousands, so knowledge doesn't drop after 65.
