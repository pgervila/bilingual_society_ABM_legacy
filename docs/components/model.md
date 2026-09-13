# `model.py` — `BiLangModel`

The central orchestrator. Owns the grid, schedule, four collaborators
(`GeoMapper`, `NetworkBuilder`, `DataProcessor`, `DataViz`), the runtime CDFs,
and the conversation engine. As of the modernization it is a **plain class**
(no longer subclasses Mesa `Model`).

## Class constants (`model.py:59-77`)

| Constant | Value | Meaning |
|---|---|---|
| `ic_pct_keys` | `[10, 25, 50, 75, 90]` | Legacy IC exposure buckets. **Currently unused** — ICs are null (see below). |
| `family_size` | `4` | Agents per initial family (2 parents + 2 children). |
| `school_lang_policy` | `[0, 1]` | Languages a school may teach in. |
| `jobs_lang_policy` / `media_lang_policy` | `None` | Placeholders. |
| `steps_per_year` | `36` | One step ≈ 10 days. |
| `max_life_steps` / `max_lifetime` | `3600` | Unified (100 years). All per-age arrays and demography curves are sized to this. |
| `langs` | `('L1', 'L12', 'L21', 'L2')` | The four per-agent memory stores. |
| `similarity_corr` | `{'L1':'L2','L2':'L1','L12':'L2','L21':'L1'}` | Which "other language" each store maps to for edit-distance word recognition. |
| `num_words_conv` | `{'VS':1,'S':3,'M':10,'L':100}` | Compressed tokens per conversation by length class. |

`_Decorators.conv_counter` is a nested decorator that increments each agent's
`_conv_counts_per_step` (used for data collection).

## `__init__` setup sequence (`model.py:79-155`)

Key parameters: `num_people`, `spoken_only=True` (→ `vocab_red` 500 vs 1000),
`width=100, height=100`, `max_people_factor=5` (caps total agent id pool),
`init_lang_distrib=[0.25, 0.65, 0.1]` (fraction L1-mono / bilingual / L2-mono),
`num_clusters=10`, `immigration=False`, `pct_immigration=0.005`,
`lang_ags_sorted_by_dist`/`lang_ags_sorted_in_clust` (spatial sorting of langs),
`mean_word_distance=0.3`, `warmup_steps=0`, `rand_seed`, `np_seed`.

Ordered steps:
1. Store params; set `vocab_red`.
2. Build `edit_distances` — `binomial` Levenshtein-distance proxies between L1↔L2 (`'original'`) and mixed stores (`'mixed'`), length `vocab_red`.
3. `set_available_ids` = pool of `max_people_factor * num_people` unique ids.
4. **`_build_cdfs()`** — generate age-indexed Zipf-Mandelbrot CDFs at runtime (replaces the dropped `lang_cdfs_vs_step.h5`). Result in `self.cdf_data = {'s': (3600, vocab_red)}`.
5. `self.lang_ICs = None` — **legacy IC file dropped; agents seeded null.**
6. `init_mode = True`.
7. Create `MultiGrid(height, width, False)` and `StagedActivationModif` with stages `["stage_1".."stage_4"]`, `shuffle=True`.
8. `GeoMapper(self, num_clusters).map_model_objects()` — lays out city + population.
9. `NetworkBuilder(self).build_networks()`.
10. `DataProcessor(self)` + `DataViz(self)`.
11. `set_conv_length_age_factor()` + `set_death_prob_curve()`.
12. `init_mode = False`.
13. Optional `check_model_set_up()`.
14. **Warmup:** run `schedule.step()` `warmup_steps` times *without* data collection (so a realistic demography emerges from null ICs before collection begins).

> **Current IC status:** `set_lang_ics` (in `agent.py`) routes every store through
> `_set_null_lang_attrs` — all agents start knowing nothing. `warmup_steps` is the
> current mechanism for reaching a realistic state. Restoring legacy graded ICs is
> deferred (see `scripts/build_ics.py`, a WIP reproduction, and the follow-up plan).

## `_build_cdfs` (`model.py:157-171`)

For each age `0..3599`, `n = max(vocab_ceiling_curve(age), vocab_red)` gives the
raw vocabulary ceiling, then `Zipf_Mand_3S_CDF_comp(n, n_red=vocab_red)` compresses
it to a `vocab_red`-length CDF. `cdf_data['s'][age]` is consumed by `randZipf`
in `pick_vocab`/`study_vocab` and by `len()` in pct-knowledge calcs.

## Demography curves

- **`set_conv_length_age_factor(age_1=14, age_2=65, ...)`** (`model.py:188`): a 3-section curve (S-curve rise in childhood → plateau at 1 → exponential decay after 65), stored inverted as `conv_length_age_factor` (indexed by age-step). Modulates how many words agents utter by age.
- **`set_death_prob_curve(a,b,c)`** (`model.py:225`): fitted Gompertz-type per-step death probability (`de Beer / TOPALS`; life expectancy ≈77y, std ≈15y), divided by `steps_per_year`.

## `get_newborn_lang(parent1, parent2)` (`model.py:240`)

Static. Decides the newborn's language type and which language each parent speaks
to it, from the parents' L1/L2 pct knowledge. **Modernization fix:** normalizes the
probability vector with a uniform fallback when `pcs.sum() == 0` (avoids a
divide-by-zero under null ICs).

## The conversation engine

### `run_conversation(ag_init, others, bystander=None, def_conv_length='M', num_days=10)` (`model.py:280`)

1. Build `ags = [ag_init, *others]`.
2. `conv_params = get_conv_params(ags, ...)` (may raise `ZeroDivisionError` → conversation aborted).
3. For each agent + its assigned `lang`: if the agent is **not** the `mute_type`, it `pick_vocab(...)` (speaks); every other agent (plus optional `bystander`) `update_lang_arrays(spoken_words, mode_type='listen', delta_s_factor=0.1)`.
4. Mute agents call `update_lang_exclusion()` instead.
5. Acquaintance edges updated in `known_people_network` via `update_acquaintances`.

The **speak (ds_factor=1) vs listen (ds_factor=0.1) asymmetry** is applied here: speakers reinforce memory ~10× more than listeners.

### `get_conv_params(ags, def_conv_length='M')` (`model.py:331`) — MAXIMIN

Implements Van Parijs's MAXIMIN language rule. Returns a `conv_params` dict:
`lang_group` (per-agent language, tuple), `mute_type` (language type that must stay
silent), `multilingual` (bool), `conv_length`, `min_group_age`, `fav_langs`.

Decision logic by the **set of language types present** (`ags_lang_types`):
- `{0}` or `{0,1}` → group speaks L1 (`compute_lang_group(0)`).
- `{1,2}` or `{2}` → group speaks L2 (`compute_lang_group(1)`).
- `{1}` (all bilingual) → initiator's dominant language.
- `{0,2}` or `{0,1,2}` (monolinguals on both sides): sub-cases based on who is a *real* monolingual (below `lang_thresholds['understand']` in the other language):
  - Nobody truly monolingual → each agent speaks their favourite; `multilingual=True`, `conv_length='S'`.
  - Real L1-monolinguals only → conversation in L1, L2-mono agents muted (`conv_length='VS'`).
  - Real L2-monolinguals only → symmetric.
  - Both sides truly monolingual → initiator speaks to whoever understands; majority-language pick if initiator is bilingual; the other side listens without understanding (`conv_length='VS'`).

`compute_lang_group` biases toward the *average language previously spoken* with
already-known interlocutors (relationship-specific stickiness via
`known_people_network[ag_init][ag]['lang']`), falling back to a default for unknowns.

## `run_model(steps, save_data_freq=50, pickle_model_freq=5000, viz_steps_period=None, save_dir='')` (`model.py:742`)

Sets `save_dir` on both the model and the `data_process`, then loops `steps` times:
`step()` (= `schedule.step()` + `data_process.collect()`), and periodically
`save_model_data` (Parquet parts), `pickle_model` (dill), and viz frames.
Progress bar via `tqdm`.

## `update_centers()` (`model.py:716`)

Called yearly by the scheduler. Two phases across all clusters:
**phase 1** (`update_courses_phase_1` on faculties + schools; `set_lang_policy` on jobs),
then **phase 2** (`update_courses_phase_2`; every 4 years also `swap_teachers_courses`).
See [city_objects.md](city_objects.md).

Other model methods: `set_lang_ics_in_family` / `get_lang_fam_members` (family IC
setup, currently null), `add_new_agent_to_model`, `update_friendships`,
`add_immigration_family` / `add_immigration`, `remove_from_locations` /
`remove_after_death` (deregister agents from grid, schedule, networks, city objects).
