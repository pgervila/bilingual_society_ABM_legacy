# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`bilangsim` is an agent-based model (ABM) that simulates the long-term evolution of language use in bilingual societies, intended as a tool to study the sustainability of minority languages. Agents are individuals who age, form families, attend school, get jobs, build social networks, converse, and learn/forget languages word-by-word using an Ebbinghaus/SM-2 memory model. Bilingualism and code-switching are **emergent** properties of conversation dynamics, not programmed behaviors.

This is a **legacy codebase** started years ago and recently resumed. It does **not currently run** out of the box (see "Current status / known blockers" below). The immediate goal is documentation, tech-stack modernization, and bug-fixing **before** any model changes.

## Running the model

Requires the `PYTHONHASHSEED` env var to be set (the model reads `os.environ['PYTHONHASHSEED']` directly at import time in `model.py` and will `KeyError` without it):

```bash
export PYTHONHASHSEED=0
python -c "from bilangsim import BiLangModel; m = BiLangModel(400, num_clusters=2); m.run_model(5)"
```

Install (editable): `pip install -e .`

## Running tests

Tests use `pytest` and live in `tests/`. There is no `conftest.py`, `pytest.ini`, or other pytest config.

```bash
export PYTHONHASHSEED=0
pytest tests/                          # all tests
pytest tests/test_model.py             # one file
pytest tests/test_agent.py::test_name  # one test
```

Note: tests construct a real `BiLangModel`, so they inherit the HDF5-data blocker below.

## Current status / known blockers

These are the things standing between the repo and a working simulation. They are the expected focus of upcoming work.

1. **Missing HDF5 data files.** `BiLangModel.__init__` (`model.py:114-115`) loads two `.h5` files via `deepdish` (`bilangsim/data/init_conds/lang_spoken_ics_vs_step.h5` and `bilangsim/data/cdfs/lang_cdfs_vs_step.h5`). Neither the files nor the `bilangsim/data/` directory exist in the repo — `dd.io.load` raises `OSError` immediately. See `docs/legacy_hdf5_data.md` for the file schemas, how to regenerate them, and recommended replacement storage formats (Zarr / NPZ / runtime generation).
2. **Stale 2018-era tech stack.** `requirements.txt` pins `numpy==1.15.0`, `pandas==0.23.4`, `mesa==0.7.8.1`, `numba==0.40.0`, `Cython==0.28.5`, `tables==3.4.4`, `deepdish==0.3.6`, etc. — none install cleanly on a modern Python. `requirements.txt` also contains a self-referencing `-e git+https://...#egg=bilangsim` line that must be removed. `setup.py` declares `python_requires='>=3.6'`.
3. **Mesa API coupling.** The code subclasses `mesa.Model`, `mesa.time.StagedActivation` (see `schedule.py`), uses `mesa.space.MultiGrid`, and `DataProcessor` subclasses `mesa`'s `DataCollector`. Mesa's API changed substantially after 0.7.8.1, so a version bump will require code changes.
4. **`deepdish` is unmaintained** and pulls in `tables`/HDF5. Both files it loads are read-only config; `docs/legacy_hdf5_data.md` recommends dropping the dependency entirely.

## Architecture

### Orchestration

`BiLangModel` (`model.py`) is the central object. Its `__init__` wires together four collaborators and is responsible for the whole setup sequence:

- **`GeoMapper`** (`geomapping.py`) — lays out the city: computes cluster centers/sizes on the grid, places `Job`s, `School`s, `University`s/`Faculty`s, `Home`s, then creates the initial population of agents and families and distributes them by language across clusters. Also handles runtime spatial bookkeeping (`update_agent_clust_info`, adding agents to grid/schedule).
- **`NetworkBuilder`** (`networks.py`) — owns four `networkx` graphs: `known_people_network` (DiGraph, edges labeled with spoken language), `friendship_network` (Graph), `family_network` (DiGraph, edges labeled with kinship), `jobs_network` (Graph). Also computes per-step adjacency matrices.
- **`StagedActivationModif`** (`schedule.py`) — the scheduler (subclass of Mesa `StagedActivation`).
- **`DataProcessor` / `DataViz`** (`dataprocess.py`) — data collection, persistence (pickling the model, saving HDF5 data), plotting, and `PostProcessor` for offline analysis.

`BiLangModel` has an `init_mode` flag that is `True` during construction and `False` during simulation — several agent/object methods branch on `model.init_mode`.

### The simulation step (4 stages)

`StagedActivationModif.step()` (`schedule.py`) runs once per step. There are **36 steps per simulated year** (`steps_per_year`); `max_lifetime = 4000` steps. Each step:

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

- **Determinism** depends on `PYTHONHASHSEED=0` plus the `random`/`numpy` seeds chosen in `model.py:29-35` (currently randomized each run; hard-coded seed lines are commented out).
- Language integers throughout the code: `0` = monolingual L1 ("spa"), `1` = bilingual, `2` = monolingual L2 ("cat"). Older variable/label names use `spa`/`cat`/`bil`.
- Lots of `# TODO` comments mark known-incomplete logic — they are original-author notes, not necessarily things to fix now.
- The repo has many remote branches (`git branch -a`) representing historical feature work; `development` is the working branch, `master` is the main branch.
- `docs/` was written in a prior session and is current as of May 2026: `legacy_language_model_analysis.md` (the language model) and `legacy_hdf5_data.md` (the HDF5 files and storage-format options).
