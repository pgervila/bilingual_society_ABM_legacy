# bilangsim Tech-Stack Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the abandoned `bilangsim` agent-based model running cleanly on a modern Python stack with modern storage, without changing any model science.

**Architecture:** Six sequential, independently-committable phases. **Revised 2026-05-14:** the original plan assumed the 2018 dependency stack could be installed and used as a frozen regression baseline through Phases 1–4, with the version jump deferred to last. That is not possible — the pinned 2018 packages (`numpy==1.15.0`, `Cython==0.28.5`, `tables==3.4.4`, `pandas==0.23.4`, `mesa==0.7.8.1`) have no wheels for Python ≥3.8 and will not build on a modern toolchain, and no environment with them exists. So the environment jump moves to the **front** (Phase 0), and all code work happens against the modern stack. The new sequencing: stand up the modern env → make the package import → make the model construct & run → green test suite + frozen baseline → warmup/constants cleanup → storage to Parquet → final verification.

**Tech Stack:** Python 3.13, numpy 2.x, pandas 2.x, networkx 3.x, numba (latest compatible), pyarrow (Parquet), dill (snapshots), tqdm, uv (dependency management). Mesa, deepdish, tables, pyprind are removed entirely.

---

## Context

`bilangsim` simulates long-term language evolution in bilingual societies. It was started years ago, abandoned, and is now being revived. **It does not currently run at all** — `BiLangModel.__init__` calls `deepdish.io.load` on two HDF5 data files (`bilangsim/data/init_conds/lang_spoken_ics_vs_step.h5` and `bilangsim/data/cdfs/lang_cdfs_vs_step.h5`) that do not exist in the repo. The test suite constructs a real model, so the tests are dead too. The stack is frozen at 2018 and `requirements.txt` contains a broken self-referencing `-e git+...#egg=bilangsim` line.

Decisions made by the project owner that shaped this plan:
1. **Drop Mesa entirely** — the project subclasses and fully overrides `Model`, `StagedActivation`, `MultiGrid`, and `DataCollector`, and its agents aren't even Mesa agents. Vendoring ~250 lines of owned code is lower-risk than chasing Mesa's fast-moving API.
2. **Initial conditions: null ICs + warmup phase** — drop the ~520 MB IC file (its generation recipe isn't in the repo); seed all agents with the existing `_set_null_lang_attrs` and add a configurable warmup-steps phase.
3. **CDFs: generate at runtime** — drop the CDF file; compute age-indexed Zipf-Mandelbrot CDFs at construction from the existing `zipf_generator/Zipf.py` math.
4. **Results storage: Parquet, keep dill snapshots** — replace the pandas-HDFStore output with Parquet; keep `dill` for full-model snapshots.
5. **Scope: stop at a clean run** — fix only bugs that block a clean run with passing tests. Documented model bugs (e.g. the L12 `pct` bug) and model improvements are deferred to a later plan.
6. **Environment via `uv`** — there is no bare `python` on PATH; use `uv python list`, `uv sync`, `uv run`. Python 3.13.12 is already installed locally.

**Intended outcome:** `uv run pytest tests/` passes on Python 3.13, and `uv run python -c "from bilangsim import BiLangModel; BiLangModel(400, num_clusters=2).run_model(5)"` succeeds, with no Mesa / deepdish / tables / pyprind / HDF5 dependency anywhere.

**Working branch:** `development`. Commit after every task.

---

## Critical files

- `bilangsim/model.py` — `BiLangModel`; mesa/deepdish/pyprind imports, import-time crashes, HDF5 loads, needs `_build_cdfs`, warmup param.
- `bilangsim/agent.py` — `np.bool`/`np.float` aliases, IC setup (`set_lang_ics`, `_set_lang_attrs`, `_set_null_lang_attrs`), 6 numba `@njit` helpers.
- `bilangsim/schedule.py` — `StagedActivationModif`, subclasses `mesa.time.StagedActivation`.
- `bilangsim/dataprocess.py` — `DataProcessor` (subclasses `DataCollector`), all HDF5 output, `deepdish` reads, `matplotlib.pylab`.
- `bilangsim/networks.py` — `sklearn` imports that may have moved in modern sklearn.
- `bilangsim/zipf_generator/Zipf.py` — CDF math; needs a new `vocab_ceiling_curve`.
- `bilangsim/space.py` — **new file**, vendored MultiGrid.
- `pyproject.toml` — **new file**, replaces `setup.py` + `requirements.txt`.

---

## Phase 0 — Modern environment

**Milestone:** `uv sync --python 3.13` resolves cleanly; `setup.py` and `requirements.txt` are gone, replaced by `pyproject.toml`.

### Task 0.1: Create pyproject.toml, delete legacy packaging, sync

**Files:**
- Create: `pyproject.toml`
- Delete: `setup.py`, `requirements.txt`, `MANIFEST.in`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "bilangsim"
version = "0.2.0"
description = "Agent-based simulator of bilingual societies"
authors = [{ name = "Paolo Gervasoni Vila", email = "pgervila@gmail.com" }]
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.13",
    "pandas>=2.2",
    "numba>=0.60",
    "scikit-learn>=1.5",
    "matplotlib>=3.9",
    "networkx>=3.3",
    "pyarrow>=17.0",
    "dill>=0.3.8",
    "tqdm>=4.66",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-repeat>=0.9",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["bilangsim"]
```

Delete `setup.py`, `requirements.txt` (removes the broken `-e git+...` line), and `MANIFEST.in` (it only referenced the dropped `data/*.h5` files).

- [ ] **Step 2: Sync the environment**

Run: `uv sync --python 3.13`
Expected: a `.venv` is created and all dependencies resolve. If `numba` cannot resolve against the newest `numpy`, cap numpy to numba's supported range (e.g. `"numpy>=2.0,<2.3"`) and re-run — record the chosen pin in a comment in `pyproject.toml`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git rm setup.py requirements.txt MANIFEST.in
git commit -m "build: migrate to uv + pyproject.toml on Python 3.13, drop legacy packaging"
```

**Phase 0 verification:** `uv run python --version` reports 3.13.x; `uv pip list` shows the modern stack with no mesa/deepdish/tables/pyprind.

---

## Phase 1 — Make the package import cleanly

**Milestone:** `uv run python -c "import bilangsim"` succeeds; `grep -rn 'mesa\|pyprind' bilangsim/` is empty. (Construction will still fail — that is Phase 2.)

### Task 1.1: Fix import-time crash in model.py

**Files:**
- Modify: `bilangsim/model.py:3`, `bilangsim/model.py:39`

- [ ] **Step 1: Remove the dead `__future__` import and fix the `PYTHONHASHSEED` access**

In `bilangsim/model.py`, delete line 3 (`from __future__ import division`). Change line 39 from `print('python hash seed is', os.environ['PYTHONHASHSEED'])` to:

```python
print('python hash seed is', os.environ.get('PYTHONHASHSEED', 'not set'))
```

- [ ] **Step 2: Commit**

```bash
git add bilangsim/model.py
git commit -m "fix: guard PYTHONHASHSEED access and drop dead __future__ import"
```

### Task 1.2: Vendor a minimal MultiGrid

**Files:**
- Create: `bilangsim/space.py`
- Test: `tests/test_space.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_space.py
from bilangsim.space import MultiGrid


class _Ag:
    pos = None


def test_place_move_remove_and_contents():
    grid = MultiGrid(10, 10, False)
    a, b = _Ag(), _Ag()
    grid.place_agent(a, (2, 3))
    grid.place_agent(b, (2, 3))
    assert set(grid.get_cell_list_contents((2, 3))) == {a, b}
    assert a.pos == (2, 3)
    grid.move_agent(a, (4, 5))
    assert grid.get_cell_list_contents((2, 3)) == [b]
    assert set(grid.get_cell_list_contents((4, 5))) == {a}
    grid._remove_agent((4, 5), a)
    assert grid.get_cell_list_contents((4, 5)) == []
    assert a.pos is None
    assert grid.width == 10 and grid.height == 10
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_space.py -v`
Expected: FAIL — `ModuleNotFoundError: bilangsim.space`.

- [ ] **Step 3: Write `bilangsim/space.py`**

```python
"""Minimal grid replacement for the dropped Mesa dependency.

Provides only the MultiGrid surface bilangsim actually uses:
place_agent, move_agent, _remove_agent, get_cell_list_contents, width, height.
"""


class MultiGrid:
    """A 2D grid where each cell holds a set of agents.

    Signature is (width, height, torus) to match mesa.space.MultiGrid. Note that
    model.py historically calls MultiGrid(height, width, False) with
    width == height == 100, so the argument swap is harmless and is kept as-is.
    """

    def __init__(self, width, height, torus=False):
        self.width = width
        self.height = height
        self.torus = torus
        self._cells = {}

    def _cell(self, pos):
        if pos not in self._cells:
            self._cells[pos] = set()
        return self._cells[pos]

    def place_agent(self, agent, pos):
        self._cell(pos).add(agent)
        agent.pos = pos

    def move_agent(self, agent, pos):
        old = getattr(agent, 'pos', None)
        if old is not None and old in self._cells:
            self._cells[old].discard(agent)
        self.place_agent(agent, pos)

    def _remove_agent(self, pos, agent):
        if pos in self._cells:
            self._cells[pos].discard(agent)
        if getattr(agent, 'pos', None) == pos:
            agent.pos = None

    def get_cell_list_contents(self, cell_list):
        """Accept a single (x, y) position or an iterable of positions."""
        if (len(cell_list) == 2 and
                all(isinstance(c, int) for c in cell_list)):
            cell_list = [cell_list]
        contents = []
        for pos in cell_list:
            contents.extend(self._cell(pos))
        return contents
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_space.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bilangsim/space.py tests/test_space.py
git commit -m "feat: vendor minimal MultiGrid to replace mesa.space"
```

### Task 1.3: Make the scheduler standalone

**Files:**
- Modify: `bilangsim/schedule.py`

- [ ] **Step 1: Replace the Mesa import and add the inherited bookkeeping**

In `bilangsim/schedule.py`, replace lines 1–8 (the imports and class header) with:

```python
import random

from .agent import IndepAgent, Young


class StagedActivationModif:
    """Standalone staged-activation scheduler.

    Formerly subclassed mesa.time.StagedActivation, but bilangsim fully overrides
    step(); only the trivial bookkeeping from the Mesa base class is reproduced here.
    """

    def __init__(self, model, stage_list, shuffle=False, shuffle_between_stages=False):
        self.model = model
        self.steps = 0
        self.time = 0
        self.agents = []
        self.stage_list = stage_list
        self.shuffle = shuffle
        self.shuffle_between_stages = shuffle_between_stages
        self.stage_time = 1 / len(stage_list)

    def add(self, agent):
        self.agents.append(agent)

    def remove(self, agent):
        while agent in self.agents:
            self.agents.remove(agent)

    def get_agent_count(self):
        return len(self.agents)
```

The existing `step(self, pct_threshold=0.9)` and `replace_agent(self, old_agent, new_agent)` methods stay exactly as they are — they already don't call `super()`.

- [ ] **Step 2: Verify**

Run: `grep -n 'mesa' bilangsim/schedule.py`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add bilangsim/schedule.py
git commit -m "refactor: make StagedActivationModif standalone, drop mesa.time"
```

### Task 1.4: Make DataProcessor standalone

**Files:**
- Modify: `bilangsim/dataprocess.py:5`, `bilangsim/dataprocess.py:10`, `bilangsim/dataprocess.py:15-40`

- [ ] **Step 1: Fix imports and inline the DataCollector storage logic**

In `bilangsim/dataprocess.py`: change `import matplotlib.pylab as plt` (line 5) to `import matplotlib.pyplot as plt`. Delete line 10 (`from mesa.datacollection import DataCollector`). Change the class declaration on line 15 from `class DataProcessor(DataCollector):` to `class DataProcessor:`.

Replace the `super().__init__(model_reporters={...}, agent_reporters={...})` call (lines 19–40) with direct attribute assignment — keep the **exact same reporter dicts**, just bind them and initialize the storage dicts:

```python
        self.model_reporters = {"pct_spa": lambda dp: dp.get_lang_stats(0),
                                "pct_bil": lambda dp: dp.get_lang_stats(1),
                                "pct_cat": lambda dp: dp.get_lang_stats(2),
                                "total_num_agents": lambda dp: len(dp.model.schedule.agents),
                                "pct_cat_in_biling": lambda dp: dp.get_global_bilang_inner_evol()
                                }
        self.agent_reporters = {"pct_cat_knowledge": lambda a: a.lang_stats['L2']['pct'][a.info['age']],
                                "pct_L21_knowledge": lambda a: a.lang_stats['L21']['R'].mean(),
                                "pct_spa_knowledge": lambda a: a.lang_stats['L1']['pct'][a.info['age']],
                                "pct_L12_knowledge": lambda a: a.lang_stats['L12']['R'].mean(),
                                "tokens_per_step_spa": lambda a: (a.wc_final['L1'] - a.wc_init['L1']).sum(),
                                "tokens_per_step_cat": lambda a: (a.wc_final['L2'] - a.wc_init['L2']).sum(),
                                "x": lambda a: a.pos[0],
                                "y": lambda a: a.pos[1],
                                "age": lambda a: a.info['age'],
                                "language": lambda a: a.info['language'],
                                "excl_c": lambda a: a.lang_stats['L1']['excl_c'][a.info['age']] if a.info['language'] == 2 else a.lang_stats['L2']['excl_c'][a.info['age']],
                                "clust_id": lambda a: a.loc_info['home'].info['clust'],
                                "agent_type": lambda a: type(a).__name__,
                                "num_conv_step": lambda a: a._conv_counts_per_step
                                }
        self.model_vars = {k: [] for k in self.model_reporters}
        self.agent_vars = {k: [] for k in self.agent_reporters}
```

(`collect`, `get_model_vars_dataframe`, `get_agent_vars_dataframe` already override the Mesa versions — leave them. The `import deepdish as dd` line stays for now; it is removed in Phase 5 when its remaining uses are ported.)

- [ ] **Step 2: Verify**

Run: `grep -n 'mesa' bilangsim/dataprocess.py`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add bilangsim/dataprocess.py
git commit -m "refactor: make DataProcessor standalone, drop mesa.datacollection"
```

### Task 1.5: Drop mesa/pyprind from model.py, fix matplotlib, make package import

**Files:**
- Modify: `bilangsim/model.py` (imports lines 12–19, `run_model` ~714–741), and any import-blocker the verification surfaces (e.g. `bilangsim/networks.py` sklearn imports)

- [ ] **Step 1: Fix model.py imports and the progress bar**

In `bilangsim/model.py`:
- Line 12: change `import matplotlib.pylab as plt` → `import matplotlib.pyplot as plt`.
- Line 14: delete `import pyprind`; add `from tqdm import tqdm` near the other imports.
- Lines 18–19: delete `from mesa import Model` and `from mesa.space import MultiGrid`; add `from .space import MultiGrid` next to `from .schedule import StagedActivationModif`.
- Change `class BiLangModel(Model):` to `class BiLangModel:`.
- In `run_model`, replace the `pyprind` progress bar: delete the `pbar = pyprind.ProgBar(steps)` line and the `pbar.update()` line, and change the loop `for _ in range(steps):` to `for _ in tqdm(range(steps)):`.

- [ ] **Step 2: Confirm nothing relied on the Mesa `Model` base**

Run: `grep -n 'super(\|self\.running\|self\._seed' bilangsim/model.py`
Expected: no output. If `self.running` *is* found, add `self.running = True` in `__init__`.

- [ ] **Step 3: Make the package import**

Run: `uv run python -c "import bilangsim"`
Expected: succeeds. If it fails on an unrelated import (most likely `bilangsim/networks.py` doing `from sklearn.utils import DataConversionWarning` — in modern sklearn this is `from sklearn.exceptions import DataConversionWarning`), fix that import at its site and re-run until the package imports cleanly.

- [ ] **Step 4: Verify mesa and pyprind are gone**

Run: `grep -rn 'mesa\|pyprind' bilangsim/`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add bilangsim/
git commit -m "refactor: drop mesa and pyprind from model, package now imports on modern stack"
```

**Phase 1 verification:** `uv run python -c "import bilangsim"` succeeds; `grep -rn 'mesa\|pyprind' bilangsim/` is empty; `uv run pytest tests/test_space.py` passes.

---

## Phase 2 — Make the model construct and run

**Milestone:** `tests/test_smoke.py` passes — `BiLangModel(...)` constructs and `run_model(3)` runs.

### Task 2.1: Add the smoke test (the failing test)

**Files:**
- Create: `tests/test_smoke.py` (already drafted in the working tree)

- [ ] **Step 1: Confirm the smoke test content**

```python
# tests/test_smoke.py
import os
os.environ.setdefault("PYTHONHASHSEED", "0")

from bilangsim import BiLangModel


def test_model_constructs():
    model = BiLangModel(200, num_clusters=1)
    assert model.schedule.get_agent_count() > 0


def test_model_runs_short():
    model = BiLangModel(200, num_clusters=1)
    model.run_model(3)
    assert model.schedule.steps == 3
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `OSError` from the `dd.io.load` calls (missing HDF5 files) during construction, or an `AttributeError` from `np.bool`/`np.float`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add smoke test for model construction and short run"
```

### Task 2.2: Replace deprecated numpy dtype aliases

**Files:**
- Modify: `bilangsim/agent.py` (and any other file the grep finds)

- [ ] **Step 1: Find every deprecated alias**

Run: `grep -rn 'np\.bool\b\|np\.float\b\|np\.int\b\|np\.object\b' bilangsim/`
Expected: matches in `agent.py` (at least `dtype=np.bool` ~line 97, `dtype=np.float` ~lines 153–154, and `reset_step_mask` ~line 757).

- [ ] **Step 2: Replace each with the Python builtin**

For every match, replace `np.bool` → `bool`, `np.float` → `float`, `np.int` → `int`, `np.object` → `object`. These aliases were removed in numpy 2.0; the builtins are the correct replacement.

- [ ] **Step 3: Verify no aliases remain**

Run: `grep -rn 'np\.bool\b\|np\.float\b\|np\.int\b\|np\.object\b' bilangsim/`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add bilangsim/agent.py
git commit -m "fix: replace removed numpy dtype aliases with builtins"
```

### Task 2.3: Add the age→vocab-ceiling curve

**Files:**
- Modify: `bilangsim/zipf_generator/Zipf.py`, `bilangsim/zipf_generator/__init__.py`

- [ ] **Step 1: Add `vocab_ceiling_curve` to `Zipf.py`**

Append to `bilangsim/zipf_generator/Zipf.py`:

```python
def vocab_ceiling_curve(age_steps, steps_per_year=36, n_min=500, n_max=10000,
                        midpoint_years=8, rate=0.4):
    """Age-dependent ceiling on the raw vocabulary size an agent can sample from.

    Logistic growth in years: from n_min in early childhood to n_max in adulthood.
    Replaces the legacy externally-precomputed age->vocab mapping. Returns an int
    (or int array) suitable as the `n` argument to the Zipf CDF generators.
    """
    age_years = np.asarray(age_steps) / steps_per_year
    n = n_min + (n_max - n_min) / (1 + np.exp(-rate * (age_years - midpoint_years)))
    return np.rint(n).astype(np.int64)
```

- [ ] **Step 2: Export it**

In `bilangsim/zipf_generator/__init__.py`, update to:

```python
from .Zipf import Zipf_CDF, Zipf_Mandelbrot_CDF, randZipf
from .Zipf import Zipf_CDF_compressed, Zipf_Mand_CDF_compressed
from .Zipf import Zipf_Mand_3S_CDF_comp, vocab_ceiling_curve
```

- [ ] **Step 3: Smoke-check the curve**

Run: `uv run python -c "from bilangsim.zipf_generator import vocab_ceiling_curve as v; print(v(0), v(8*36), v(3599))"`
Expected: three increasing integers, the first ≈ 500, the last close to 10000.

- [ ] **Step 4: Commit**

```bash
git add bilangsim/zipf_generator/Zipf.py bilangsim/zipf_generator/__init__.py
git commit -m "feat: add age-dependent vocab-ceiling curve for runtime CDF generation"
```

### Task 2.4: Generate CDFs at runtime, remove the CDF HDF5 load

**Files:**
- Modify: `bilangsim/model.py` (imports near line 15, `__init__` near lines 113–115, add `_build_cdfs` and `max_life_steps` class attr)

- [ ] **Step 1: Add `_build_cdfs` and call it from `__init__`**

In `bilangsim/model.py`, remove `import deepdish as dd` (line 15). Remove the two `dd.io.load` lines (114–115). In their place in `__init__`:

```python
        self._build_cdfs()
        self.lang_ICs = None  # legacy IC file dropped; agents use null ICs (see set_lang_ics)
```

Add `max_life_steps = 3600` as a `BiLangModel` class attribute next to `max_lifetime` (Phase 4 reconciles the 3600/4000 split). Define the method on `BiLangModel`:

```python
    def _build_cdfs(self):
        """Build age-indexed Zipf-Mandelbrot CDFs at construction time.

        Replaces the legacy lang_cdfs_vs_step.h5 file. cdf_data['s'][age] is the
        CDF over compressed word indices for an agent of `age` steps, consumed by
        randZipf() in pick_vocab/study_vocab and by len() in the pct-knowledge calc.
        """
        from .zipf_generator.Zipf import Zipf_Mand_3S_CDF_comp, vocab_ceiling_curve
        n_ages = self.max_life_steps
        cdfs = np.empty((n_ages, self.vocab_red), dtype=np.float64)
        for age in range(n_ages):
            n = max(int(vocab_ceiling_curve(age, steps_per_year=self.steps_per_year)),
                    self.vocab_red)
            cdfs[age] = Zipf_Mand_3S_CDF_comp(n, n_red=self.vocab_red)
        self.cdf_data = {'s': cdfs}
```

- [ ] **Step 2: Verify CDF math in isolation**

Run: `uv run python -c "import numpy as np; from bilangsim.zipf_generator.Zipf import Zipf_Mand_3S_CDF_comp as f; c=f(2000,n_red=500); print(c.shape, c.min()>=0, c.max()<=1.0+1e-9, np.all(np.diff(c)>=-1e-12))"`
Expected: `(500,) True True True`.

- [ ] **Step 3: Commit**

```bash
git add bilangsim/model.py
git commit -m "feat: generate age-indexed Zipf CDFs at runtime, drop CDF HDF5 load"
```

### Task 2.5: Route all initial conditions through null ICs, get the model running

**Files:**
- Modify: `bilangsim/agent.py` — `set_lang_ics` (~170), `_set_lang_attrs` (~114), plus any construction-blocking breakage surfaced

- [ ] **Step 1: Make `set_lang_ics` not depend on `lang_ICs`**

Replace the body of `set_lang_ics` (`agent.py:170-198`) with:

```python
    def set_lang_ics(self, s_0=0.01, t_0=1000, biling_key=None):
        """ Set agent's linguistic Initial Conditions.

        The legacy per-age IC file (lang_spoken_ics_vs_step.h5) has been dropped.
        All agents start from null linguistic knowledge; a realistic demography is
        produced by the model's warmup phase instead (see BiLangModel warmup_steps).
        `biling_key` is accepted for call-site compatibility but is currently a
        no-op — reintroducing graded initial bilingualism is deferred to the
        model-improvement plan.
        """
        for lang in ('L1', 'L2', 'L12', 'L21'):
            self._set_null_lang_attrs(lang, s_0, t_0)
        # set weights to model reaction to linguistic exclusion
        self.set_excl_weights()
```

- [ ] **Step 2: Delete the now-dead `_set_lang_attrs`**

Remove `_set_lang_attrs` entirely (`agent.py:114-143`). Verify nothing calls it: `grep -rn '_set_lang_attrs' bilangsim/` → no call sites. Leave `_set_null_lang_attrs` untouched.

- [ ] **Step 3: Run the smoke test, fix construction-blocking breakage**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS. If construction or the short run fails on a numpy 2.x / pandas 2.x / numba / networkx 3.x incompatibility, fix it at its call site and re-run until both smoke tests pass. Likely culprits: `np.NaN`→`np.nan`, `np.in1d`→`np.isin`, numba `@njit` compile errors in the 6 helpers, networkx 2→3 graph API in `networks.py`.

- [ ] **Step 4: Commit**

```bash
git add bilangsim/
git commit -m "feat: seed all agents with null ICs; model constructs and runs on modern stack"
```

**Phase 2 verification:** `uv run pytest tests/test_smoke.py -v` passes; `uv run python -c "from bilangsim import BiLangModel; BiLangModel(200, num_clusters=1).run_model(3)"` succeeds.

---

## Phase 3 — Full test suite green + frozen baseline

**Milestone:** `uv run pytest tests/` runs to completion; every failure is either fixed or recorded as an ACCEPTED deferred-model-bug; the pass/fail set is committed as the regression baseline for Phases 4–6.

### Task 3.1: Run the full suite and fix stack-migration breakage

**Files:**
- Modify: `bilangsim/agent.py`, `bilangsim/networks.py`, `bilangsim/dataprocess.py`, `bilangsim/city_objects.py`, `bilangsim/geomapping.py`, and any site the test run flags

- [ ] **Step 1: Run the full suite**

Run: `PYTHONHASHSEED=0 uv run pytest tests/ -v --tb=short`
Expected: the suite runs to completion. Catalogue every failure.

- [ ] **Step 2: Fix every stack-migration failure**

Triage each failure. **Stack-migration breakage** (numpy 2.x: `np.NaN`, `np.in1d`, copy-semantics, scalar-conversion errors; pandas 2.x: `DataFrame.append` removal, `groupby` `level=` changes, `from_dict` dtype shifts; networkx 3.x: graph/adjacency API renames in `networks.py`, especially `compute_adj_matrices`; numba: `@njit` compile errors) → **fix it now** at the call site. `grep -rn 'np\.NaN\|\.append(' bilangsim/` is a useful starting sweep (list `.append` is fine; only DataFrame/Series `.append` was removed).

- [ ] **Step 3: Commit the fixes**

```bash
git add bilangsim/
git commit -m "fix: port numpy/pandas/networkx/numba call sites to the modern stack"
```

### Task 3.2: Triage remaining failures and freeze the baseline

**Files:**
- Create: `docs/superpowers/plans/phase3-baseline.txt`

- [ ] **Step 1: Re-run the full suite and capture it**

Run: `PYTHONHASHSEED=0 uv run pytest tests/ -v --tb=short | tee docs/superpowers/plans/phase3-baseline.txt`

- [ ] **Step 2: Triage every remaining failure**

For each still-failing test: if it fails due to a **documented deferred model bug** (cross-check `docs/legacy_language_model_analysis.md`, e.g. anything touching the L12 `pct` update), annotate it ACCEPTED in the baseline file with a one-line reason. If it fails for any other reason, **fix it now** and re-run. Do not finish the phase with unexplained failures.

- [ ] **Step 3: Commit the frozen baseline**

```bash
git add docs/superpowers/plans/phase3-baseline.txt
git commit -m "test: freeze Phase 3 test baseline on the modern stack"
```

**Phase 3 verification:** `pytest tests/` runs to completion; `phase3-baseline.txt` records the pass/fail set; every failure is fixed or ACCEPTED. This set is the regression baseline for Phases 4–6.

---

## Phase 4 — Warmup phase + constant reconciliation

**Milestone:** the model can be run with a configurable warmup; the 3600/4000 age-array inconsistency is resolved; the Phase 3 baseline still holds.

### Task 4.1: Reconcile `max_life_steps` (3600) vs `max_lifetime` (4000)

**Files:**
- Modify: `bilangsim/model.py` (class attrs ~line 71), `bilangsim/agent.py:290`

- [ ] **Step 1: Unify on a single constant**

Per-age arrays (`pct`, `excl_c`, the CDF rows) are length `max_life_steps = 3600`, but `set_conv_length_age_factor`/`set_death_prob_curve` build length-`max_lifetime = 4000` arrays. An agent older than 3600 steps would `IndexError` on the per-age arrays. In `bilangsim/model.py`, set `max_lifetime = 3600` (keep the name, change the value); confirm `max_life_steps = 3600` is present (added in Task 2.4). Leave `BaseAgent.max_life_steps = 3600` as-is.

- [ ] **Step 2: Add a guard in `grow`**

Replace `BaseAgent.grow` (`agent.py:290`) with:

```python
    def grow(self, growth_inc=1):
        self.info['age'] = min(self.info['age'] + growth_inc, self.max_life_steps - 1)
```

- [ ] **Step 3: Verify**

Run: `PYTHONHASHSEED=0 uv run pytest tests/ --tb=short`
Expected: baseline unchanged.

- [ ] **Step 4: Commit**

```bash
git add bilangsim/model.py bilangsim/agent.py
git commit -m "fix: unify age-array length on 3600 steps, guard age in grow()"
```

### Task 4.2: Add a configurable warmup phase

**Files:**
- Modify: `bilangsim/model.py` — `__init__` signature and tail
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_smoke.py`:

```python
def test_warmup_runs_before_collection():
    model = BiLangModel(200, num_clusters=1, warmup_steps=5)
    model.run_model(3)
    # warmup advances the schedule but is not part of the collected run
    assert model.schedule.steps == 8
    assert len(model.data_process.model_vars['pct_bil']) == 3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONHASHSEED=0 uv run pytest tests/test_smoke.py::test_warmup_runs_before_collection -v`
Expected: FAIL — `BiLangModel` has no `warmup_steps` parameter.

- [ ] **Step 3: Implement warmup**

Add `warmup_steps=0` to the `BiLangModel.__init__` signature and store `self.warmup_steps = warmup_steps`. At the very end of `__init__` (after `self.init_mode = False`, before the optional `check_model_set_up`):

```python
        # warmup phase: advance the simulation so a realistic demography emerges
        # from null initial conditions before data collection begins
        for _ in range(self.warmup_steps):
            self.schedule.step()
```

Confirm `data_process.collect()` is only invoked from `BiLangModel.step` and not from `schedule.step` — so warmup advances the schedule without collecting.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONHASHSEED=0 uv run pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bilangsim/model.py tests/test_smoke.py
git commit -m "feat: add configurable warmup phase before data collection"
```

**Phase 4 verification:** warmup test passes; `grep -rn '\.h5\|lang_ICs\|cdfs/' bilangsim/` shows no remaining data-file dependency; baseline holds.

---

## Phase 5 — Storage: HDFStore → Parquet, keep dill

**Milestone:** no `deepdish` / `tables` / pandas-HDFStore anywhere; tabular results round-trip through Parquet; `dill` snapshots untouched; baseline holds.

### Task 5.1: Rewrite result saving to Parquet

**Files:**
- Modify: `bilangsim/dataprocess.py` — `import deepdish` (line 7), `save_model_data` (~240), `load_model_data` (~263), `optimize_data_saving_space` (~267)
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
import os
os.environ.setdefault("PYTHONHASHSEED", "0")

from bilangsim import BiLangModel
from bilangsim.dataprocess import DataProcessor


def test_parquet_results_roundtrip(tmp_path):
    model = BiLangModel(200, num_clusters=1)
    model.run_model(4, save_data_freq=2, save_dir=str(tmp_path))
    model_df, agent_df = DataProcessor.load_model_data(save_dir=str(tmp_path))
    assert model_df is not None and len(model_df) == 4
    assert agent_df is not None and len(agent_df) > 0
    assert not any(f.endswith('.h5') for f in os.listdir(tmp_path))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONHASHSEED=0 uv run pytest tests/test_storage.py -v`
Expected: FAIL — `save_model_data` still writes `.h5`, and `load_model_data` has a different signature.

- [ ] **Step 3: Rewrite the persistence methods**

In `bilangsim/dataprocess.py`, delete `import deepdish as dd` (line 7).

Replace `save_model_data` (lines 240–261) with:

```python
    def save_model_data(self, save_data_freq):
        df_model_data = self.get_model_vars_dataframe()
        df_agent_data = self.get_agent_vars_dataframe()
        part = self.model.schedule.steps // save_data_freq
        df_model_data.to_parquet(
            os.path.join(self.save_dir, f'model_data_part{part:05d}.parquet'))
        df_agent_data.to_parquet(
            os.path.join(self.save_dir, f'agent_data_part{part:05d}.parquet'))
        if self.model.schedule.steps == save_data_freq:
            # init_conds holds numpy arrays as cell values, so pickle it rather
            # than force it through Parquet's columnar typing
            with open(os.path.join(self.save_dir, 'init_conds.pkl'), 'wb') as f:
                dill.dump(self.init_conds, f)
        # empty data to avoid keeping already-saved data in RAM
        self.model_vars = {k: [] for k in self.model_vars}
        self.agent_vars = {k: [] for k in self.agent_vars}
```

Replace `load_model_data` (lines 263–265) with:

```python
    @staticmethod
    def load_model_data(save_dir=''):
        """Load all Parquet result parts written by save_model_data.

        Returns (model_df, agent_df); either may be None if no parts exist.
        """
        import glob
        model_parts = sorted(glob.glob(os.path.join(save_dir, 'model_data_part*.parquet')))
        agent_parts = sorted(glob.glob(os.path.join(save_dir, 'agent_data_part*.parquet')))
        model_df = pd.concat([pd.read_parquet(p) for p in model_parts]) if model_parts else None
        agent_df = pd.concat([pd.read_parquet(p) for p in agent_parts]) if agent_parts else None
        return model_df, agent_df
```

Delete `optimize_data_saving_space` (lines 267–275) entirely — Parquet is already compressed. Verify no caller: `grep -rn optimize_data_saving_space bilangsim/ tests/`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONHASHSEED=0 uv run pytest tests/test_storage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bilangsim/dataprocess.py tests/test_storage.py
git commit -m "feat: write simulation results as Parquet parts, drop HDFStore/deepdish"
```

### Task 5.2: Update the remaining deepdish/HDF readers

**Files:**
- Modify: `bilangsim/dataprocess.py` — `VizImpData.__init__` (~343), `PostProcessor.__init__` (~351)

- [ ] **Step 1: Rewrite `PostProcessor.__init__`**

Replace `PostProcessor.__init__` (lines 351–354+) with:

```python
    def __init__(self, data_filename=None, save_dir=''):
        self.model_data, self.agent_data = DataProcessor.load_model_data(save_dir=save_dir)
        init_path = os.path.join(save_dir, 'init_conds.pkl')
        if os.path.exists(init_path):
            with open(init_path, 'rb') as f:
                self.init_conds = dill.load(f)
```

Read the rest of `PostProcessor`'s methods and update any that referenced `self.data['...']` (the old HDFStore) to use `self.model_data` / `self.agent_data` directly. The `data_filename` argument is kept for signature compatibility but unused.

- [ ] **Step 2: Rewrite `VizImpData.__init__`**

Replace `VizImpData.__init__` (lines 343–344) with:

```python
    def __init__(self, save_dir=''):
        self.model_data, self.agent_data = DataProcessor.load_model_data(save_dir=save_dir)
```

- [ ] **Step 3: Verify deepdish and tables are fully gone**

Run: `grep -rn 'deepdish\|dd\.io\|HDFStore\|to_hdf\|read_hdf\|import tables' bilangsim/`
Expected: no output.

- [ ] **Step 4: Run the full suite against the baseline**

Run: `PYTHONHASHSEED=0 uv run pytest tests/ -v --tb=short`
Expected: baseline pass/fail set unchanged, plus `test_storage.py` passing. Fix any new regression before committing.

- [ ] **Step 5: Commit**

```bash
git add bilangsim/dataprocess.py
git commit -m "refactor: port PostProcessor and VizImpData off HDF5 to Parquet"
```

**Phase 5 verification:** `grep -rn 'deepdish\|HDFStore\|to_hdf\|read_hdf' bilangsim/` is empty; baseline holds plus `test_storage.py` passes.

---

## Phase 6 — Final verification + docs

**Milestone:** the "clean run" target is met and `CLAUDE.md` reflects the modernized state. **Stop here per scope.**

### Task 6.1: End-to-end verification

- [ ] **Step 1: End-to-end smoke run**

Run: `PYTHONHASHSEED=0 uv run python -c "from bilangsim import BiLangModel; m = BiLangModel(400, num_clusters=2); m.run_model(5)"`
Expected: completes without error.

- [ ] **Step 2: Full suite**

Run: `PYTHONHASHSEED=0 uv run pytest tests/ -v`
Expected: pass/fail set matches `docs/superpowers/plans/phase3-baseline.txt` (accepted deferred-bug failures only).

- [ ] **Step 3: Dependency-cleanliness check**

Run: `grep -rn 'mesa\|deepdish\|dd\.io\|HDFStore\|to_hdf\|pyprind\|import tables' bilangsim/ pyproject.toml`
Expected: no output.

### Task 6.2: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` — "Running the model", "Running tests", "Current status / known blockers"

- [ ] **Step 1: Rewrite the affected sections**

Update "Running the model" and "Running tests" to use `uv run`. Replace "Current status / known blockers": the HDF5/Mesa/deepdish/stale-stack blockers are resolved; describe the new state — runs on Python 3.13 via uv; ICs are null + warmup; CDFs generated at runtime; results in Parquet; RNG no longer bit-reproducible vs. the legacy stack. Keep the pointer to the deferred model-bug list in `docs/legacy_language_model_analysis.md`.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect modernized stack and storage"
```

**Phase 6 verification:** end-to-end smoke run succeeds; `pytest tests/` matches the frozen baseline; dependency-cleanliness grep is empty.

---

## Overall verification

1. `uv sync --python 3.13` resolves cleanly.
2. `PYTHONHASHSEED=0 uv run pytest tests/` runs to completion; pass/fail set matches `docs/superpowers/plans/phase3-baseline.txt`.
3. `PYTHONHASHSEED=0 uv run python -c "from bilangsim import BiLangModel; BiLangModel(400, num_clusters=2).run_model(5)"` succeeds.
4. `grep -rn 'mesa\|deepdish\|dd\.io\|HDFStore\|to_hdf\|pyprind\|import tables' bilangsim/` returns nothing.
5. A short run with `save_dir` produces `*.parquet` result parts and an `init_conds.pkl`, and `DataProcessor.load_model_data` round-trips them.
6. No model science changed: the only behavioral differences are (a) agents start from null ICs + warmup instead of the IC file, (b) CDFs are generated rather than loaded, (c) RNG streams differ on the modern numpy. All documented model bugs remain for the follow-up plan.

## Notes for the follow-up plan (out of scope here)

- The documented model bugs in `docs/legacy_language_model_analysis.md` (L12 `pct` never updated, edit-distance cognacy proxy, age-invariant memory params, etc.).
- Reintroducing graded initial bilingualism (`biling_key`) as real science rather than the current no-op, if warmup proves insufficient.
- Tuning `vocab_ceiling_curve` parameters against real lexicon-growth literature.
- Whether the warmup length should be auto-determined (run until macro-indicators stabilize) rather than a fixed parameter.
