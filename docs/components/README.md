# Component documentation

Per-component reference for `bilangsim` **as it currently functions on this
branch** (`feature/update_all_project_dependencies_to_2026`, post-modernization).
These docs describe *implemented behaviour*, not intended/legacy behaviour. Where
current behaviour differs from the legacy model or is a known caveat, it is
flagged inline.

For the scientific model narrative and the catalogue of known model bugs, see
`../legacy_language_model_analysis.md`. For the modernization history, see
`../superpowers/plans/2026-05-14-tech-stack-modernization.md`.

## Modules

| Doc | Module | What it covers |
|---|---|---|
| [model.md](model.md) | `bilangsim/model.py` | `BiLangModel` orchestration, `__init__` setup sequence, class constants, the conversation engine (`run_conversation` / `get_conv_params` = MAXIMIN), demography curves, `run_model`, `update_centers`, warmup |
| [agent.md](agent.md) | `bilangsim/agent.py` | Agent class hierarchy, per-word memory state, the learning core (`update_lang_arrays` / `update_words_memory`), `pick_vocab` (speaking + code-switch cascade), the 4-stage behaviours per concrete class, lifecycle (`evolve`, `reproduce`, `random_death`) |
| [schedule.md](schedule.md) | `bilangsim/schedule.py` | `StagedActivationModif`, the anatomy of a single `step()`, periodic hooks |
| [geomapping.md](geomapping.md) | `bilangsim/geomapping.py` | `GeoMapper`: cluster layout, object placement, initial population generation, runtime spatial bookkeeping |
| [city_objects.md](city_objects.md) | `bilangsim/city_objects.py` | `Home`, `EducationCenter`→`School`/`Faculty`, `University`, `Job`, course/teacher lifecycle |
| [networks.md](networks.md) | `bilangsim/networks.py` | `NetworkBuilder`, the four `networkx` graphs, adjacency matrices |
| [dataprocess.md](dataprocess.md) | `bilangsim/dataprocess.py` | `DataProcessor` collection + Parquet/dill persistence, `DataViz`, `PostProcessor`, `VizImpData` |
| [space.md](space.md) | `bilangsim/space.py` | Vendored minimal `MultiGrid` (Mesa replacement) |
| [zipf_generator.md](zipf_generator.md) | `bilangsim/zipf_generator/` | Word-frequency CDF math, `randZipf`, `vocab_ceiling_curve` |

## Cross-cutting conventions

- **Language integers:** `0` = monolingual L1 ("spa"), `1` = bilingual, `2` = monolingual L2 ("cat").
- **Language store labels:** `L1`, `L2` (pure), `L12`, `L21` (code-switched transition stores). `langs = ('L1', 'L12', 'L21', 'L2')`.
- **Time:** `steps_per_year = 36`; `max_life_steps = max_lifetime = 3600` steps (100 years). One step ≈ 10 days.
- **Vocabulary:** `vocab_red = 500` (spoken-only, default) or `1000` (spoken+written). Each store holds per-word arrays of this length.
- **`init_mode`:** `True` during `BiLangModel.__init__`, `False` during simulation. Many agent/object methods branch on it.
- **Determinism:** depends on `PYTHONHASHSEED=0` plus the `rand_seed`/`np_seed` chosen at module load in `model.py` (randomized each run). RNG streams are **not** comparable to the pre-modernization numpy-1.x stack.
