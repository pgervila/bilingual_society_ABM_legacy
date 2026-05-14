# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`bilangsim` is an agent-based model (ABM) that simulates the long-term evolution of language use in bilingual societies, intended as a tool to study the sustainability of minority languages. Agents are individuals who age, form families, attend school, get jobs, build social networks, converse, and learn/forget languages word-by-word using an Ebbinghaus/SM-2 memory model. Bilingualism and code-switching are **emergent** properties of conversation dynamics, not programmed behaviors.

This is a **legacy codebase** started years ago and recently resumed. As of the May 2026 modernization it runs on a current Python stack (see "Current status" below). The next phase is a follow-up plan for model-correctness fixes and improvements; **no model science has changed yet**.

## Running the model

The project is managed with `uv`. There is no bare `python` on PATH — use `uv run`. `PYTHONHASHSEED` no longer crashes the import if unset, but set it for deterministic-as-possible runs:

```bash
uv sync --python 3.13
PYTHONHASHSEED=0 uv run python -c "from bilangsim import BiLangModel; m = BiLangModel(400, num_clusters=2); m.run_model(5)"
```

`BiLangModel` accepts `warmup_steps=N` — it advances the schedule N steps before data collection so a realistic demography emerges from null initial conditions (see "Current status").

## Running tests

Tests use `pytest` and live in `tests/`. There is no `conftest.py` or `pytest.ini` (config lives in `pyproject.toml`).

```bash
PYTHONHASHSEED=0 uv run pytest tests/                          # all tests
PYTHONHASHSEED=0 uv run pytest tests/test_model.py             # one file
PYTHONHASHSEED=0 uv run pytest tests/test_agent.py::test_name  # one test
```

Current state: 86 passing, 1 skipped (`test_run_conversation` — brittle by construction, see `docs/superpowers/plans/phase3-baseline.txt`).

## Current status

The May 2026 modernization (plan + log: `docs/superpowers/plans/2026-05-14-tech-stack-modernization.md`) is complete. Summary of what changed — **stack and storage only, no model science**:

1. **Modern stack via `uv`.** Python 3.13, numpy 2.x, pandas 3.x, networkx 3.x, numba, scikit-learn, pyarrow, dill, tqdm. `pyproject.toml` replaces `setup.py`/`requirements.txt`.
2. **Mesa removed entirely.** `bilangsim/space.py` vendors a minimal `MultiGrid`; `StagedActivationModif` and `DataProcessor` are now standalone classes; `BiLangModel` is a plain class.
3. **HDF5 input files dropped.** The two missing `.h5` files are gone. Age-indexed Zipf CDFs are generated at runtime by `BiLangModel._build_cdfs()` (using `zipf_generator.vocab_ceiling_curve`). Agents are seeded with **null initial conditions** (`_set_null_lang_attrs`) instead of the per-age IC file — so a freshly-constructed model has agents that know nothing; use `warmup_steps` to reach a realistic state.
4. **Storage modernized.** Simulation results are written as numbered Parquet parts (`DataProcessor.save_model_data` / `load_model_data`); full-model snapshots still use `dill`. No `deepdish`/`tables`/HDFStore anywhere.

**Known caveats / deferred to the follow-up plan** (see `docs/legacy_language_model_analysis.md` and the plan doc's "Notes for the follow-up plan"):
- RNG streams differ from the legacy stack (numpy 2.x), so runs are not bit-reproducible against pre-modernization results.
- `set_lang_ics`' `biling_key` is currently a no-op; graded initial bilingualism was dropped with the IC file.
- `test_run_conversation` is skipped — it fishes the shared model for agents by language label and is brittle under null ICs; needs a rewrite.
- `PostProcessor.plot_population_size` uses pandas-2.x-removed `.count(level=...)` — untested analysis helper, not run-blocking.
- The documented model bugs in `docs/legacy_language_model_analysis.md` (e.g. `L12` `pct` never updated) are untouched.

## Architecture

### Orchestration

`BiLangModel` (`model.py`) is the central object. Its `__init__` wires together four collaborators and is responsible for the whole setup sequence:

- **`GeoMapper`** (`geomapping.py`) — lays out the city: computes cluster centers/sizes on the grid, places `Job`s, `School`s, `University`s/`Faculty`s, `Home`s, then creates the initial population of agents and families and distributes them by language across clusters. Also handles runtime spatial bookkeeping (`update_agent_clust_info`, adding agents to grid/schedule).
- **`NetworkBuilder`** (`networks.py`) — owns four `networkx` graphs: `known_people_network` (DiGraph, edges labeled with spoken language), `friendship_network` (Graph), `family_network` (DiGraph, edges labeled with kinship), `jobs_network` (Graph). Also computes per-step adjacency matrices.
- **`StagedActivationModif`** (`schedule.py`) — the scheduler. Standalone class (formerly subclassed Mesa `StagedActivation`); fully overrides `step()`.
- **`DataProcessor` / `DataViz`** (`dataprocess.py`) — data collection, persistence (pickling the model with `dill`, saving results as Parquet parts), plotting, and `PostProcessor` for offline analysis. `DataProcessor` is standalone (formerly subclassed Mesa `DataCollector`).

`BiLangModel` has an `init_mode` flag that is `True` during construction and `False` during simulation — several agent/object methods branch on `model.init_mode`.

### The simulation step (4 stages)

`StagedActivationModif.step()` (`schedule.py`) runs once per step. There are **36 steps per simulated year** (`steps_per_year`); per-age arrays are sized to `max_life_steps = 3600` steps (and `max_lifetime` is unified to the same value). Each step:

1. **Pre-stage bookkeeping** (all agents): reset conversation counters, `grow()` (age +1), snapshot `wc`, age word-activation elapsed times, decay memory retention (`update_memory_retention`), reset step masks, and `update_lang_switch` (reclassify mono/bilingual).
2. **Four stages** (`stage_1`..`stage_4`): each agent's stage methods are called in shuffled order. Stage behavior is **polymorphic per agent class** — e.g. `Baby` interacts with parents, `Child`/`Adolescent` go to school, `Young`/`Adult` go to jobs and look for partners, `Pensioner` gathers family. `IndepAgent` subclasses receive their schedule index `ix_agent`; others do not.
3. **Post-stage**: snapshot final `wc`; `reproduce()` and `random_death()` for eligible agents.
4. **Periodic**: friendships updated every 2 years; `update_centers()` (schools/universities/jobs roll over courses and language policies) every year; immigration families added yearly if enabled.

### Agent class hierarchy (`agent.py`, ~2240 lines)

Mixin-style hierarchy. **Capability base classes:**

- `BaseAgent` — identity, `info`/`loc_info` dicts, `lang_stats`, growth, death, family lookups, language IC setup.
- `ListenerAgent(BaseAgent)` — `listen`, `update_lang_arrays`, `process_unknown_words`, `learn_unknown_words`, `update_words_memory`, `update_memory_retention`, school registration.
- `SpeakerAgent(ListenerAgent)` — `pick_vocab` (the core speaking algorithm), `study_vocab`, `update_acquaintances`, friendship logic, language-exclusion reactions.
- `SchoolAgent(SpeakerAgent)` — school attendance / `speak_at_school`.
- `IndepAgent(SpeakerAgent)` — agents that move around the grid independently; `move_to`, `speak_to_group`, `meet_agent`.

**Concrete age/role classes** (these are what get instantiated and scheduled): `Baby`, `Child`, `Adolescent`, `Young`, `YoungUniv`, `Adult`, `Worker`, `Teacher`, `TeacherUniv`, `Pensioner`. Agents **change class as they age** via `evolve(new_class, ...)` — e.g. `Child` → `Adolescent` → `Young`/`YoungUniv` → `Adult` → `Pensioner`, and `Adult` → `Teacher`/`TeacherUniv` on hire. `evolve` rebuilds the instance as the new class and `StagedActivationModif.replace_agent` swaps it in the schedule in place.

`agent.py` also defines a handful of `numba @njit` helpers at module top (`numba_comp_delta_S`, `numba_comp_R`, etc.) for the hot memory-update math.

### City objects (`city_objects.py`)

`Home`, `EducationCenter` (base) → `School` and `Faculty`, `University` (holds `Faculty` objects), `Job`, `MeetingPoint`, `Store`/`BookStore`. `EducationCenter` is the most complex: it manages courses keyed by student age, hires/swaps teachers (pulling staff from clusters or converting `Adult`s from companies into `Teacher`s), and rolls courses over each academic year in two phases. Several city objects define `__getstate__`/`__setstate__` for pickling.

### The language learning model

This is the scientific core and is documented in depth in **`docs/legacy_language_model_analysis.md`** — read it before touching any learning/memory/conversation logic. Key points:

- Each agent's `lang_stats` has four stores: `L1`, `L2`, `L12`, `L21` (the mixed stores hold code-switched material). Each store has per-word NumPy arrays (`S` stability, `t` elapsed steps, `R` retrievability, `wc` word count, etc.) of length `vocab_red` (500 for spoken-only, 1000 for spoken+written).
- Retention follows the Ebbinghaus curve `R = exp(-k·t/S)` with `k = ln(10/9)`; activation bumps stability via an SM-2-derived formula. **Speaking reinforces ~10× more than listening** (`ds_factor` 1.0 vs 0.1).
- `pick_vocab` samples word *intent* from an age-indexed Zipf-Mandelbrot CDF, then filters by retrievability; failed retrievals **cascade** L1→L12→L2, which is where code-switching emerges.
- Conversation language is chosen by `get_conv_params` (`model.py:303`) implementing Van Parijs's **MAXIMIN** principle, with relationship-specific language stickiness via `known_people_network`.
- Word-frequency sampling math lives in `bilangsim/zipf_generator/Zipf.py` (three Zipf variants; the realistic one is a piecewise three-stage Zipf-Mandelbrot).

`docs/legacy_language_model_analysis.md` also catalogs known model weaknesses and a prioritized improvement list (including a documented bug: `L12` `pct` is never updated, `agent.py:916`). These are **deferred** until after modernization.

## Conventions and gotchas

- **Determinism** depends on `PYTHONHASHSEED=0` plus the `random`/`numpy` seeds chosen at module load in `model.py` (randomized each run; hard-coded seed lines are commented out). Note RNG streams are not comparable to the pre-modernization (numpy 1.x) stack.
- Language integers throughout the code: `0` = monolingual L1 ("spa"), `1` = bilingual, `2` = monolingual L2 ("cat"). Older variable/label names use `spa`/`cat`/`bil`.
- Lots of `# TODO` comments mark known-incomplete logic — they are original-author notes, not necessarily things to fix now.
- The repo has many remote branches (`git branch -a`) representing historical feature work; `master` is the main branch.
- Docs in `docs/`: `legacy_language_model_analysis.md` (the language model + a prioritized list of known model weaknesses), `legacy_hdf5_data.md` (the now-dropped HDF5 files, kept for historical context), and `superpowers/plans/2026-05-14-tech-stack-modernization.md` (the modernization plan and execution log).
