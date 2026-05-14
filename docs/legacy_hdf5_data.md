# HDF5 Data Files in the Legacy ABM

The legacy `bilangsim` package loads two HDF5 files at model construction time. This document covers where they live, what they hold, whether they can be regenerated, and what alternative storage formats would fit the use case better.

## 1. Where the files are referenced

```python
# legacy_project/bilingual_society_ABM/bilangsim/model.py:114-115
self.lang_ICs  = dd.io.load(os.path.join(os.path.dirname(__file__),
                                         'data', 'init_conds',
                                         'lang_spoken_ics_vs_step.h5'))
self.cdf_data  = dd.io.load(os.path.join(os.path.dirname(__file__),
                                         'data', 'cdfs',
                                         'lang_cdfs_vs_step.h5'))
```

So the *expected* on-disk layout is:

```
bilangsim/
└── data/
    ├── init_conds/
    │   └── lang_spoken_ics_vs_step.h5     # ≈ a few hundred MB
    └── cdfs/
        └── lang_cdfs_vs_step.h5           # ≈ a few MB
```

Both are declared as `package_data` in `setup.py:14` and as `recursive-include` patterns in `MANIFEST.in:1-2`, meaning they were intended to ship with the installable package.

## 2. Current status: the files are not in the repository

A `find` over `legacy_project/` confirms **no `.h5` file is present locally**, and there is no `bilangsim/data/` directory at all:

```
$ find legacy_project -name "*.h5"
(nothing)
```

This is consistent with how the project was distributed historically: the HDF5 blobs were either kept out of git (a common pattern for large binary artefacts) and shipped via PyPI/release tarballs, or they live on the upstream GitHub repository only as Git LFS objects. Either way, **the current monorepo will not run the legacy model out of the box** — `dd.io.load` will raise `OSError` at `BiLangModel.__init__`.

This isn't a blocker for the active research direction (the new `python_version/main.py` doesn't use them), but it matters if anyone wants to reproduce historical results.

## 3. What the two files contain

The schemas can be reconstructed precisely from the code that reads them.

### 3.1 `lang_cdfs_vs_step.h5` — age-indexed Zipf CDFs

This file is consumed *only* through indexing of the form `self.model.cdf_data['s'][age]` (e.g. `agent.py:137, 162, 747, 840, 888, 890`). So the schema is:

```
lang_cdfs_vs_step.h5
└── 's'                              # single top-level key ('s' likely for "spoken")
    └── <numpy array, shape (max_life_steps, n_red_at_age)>
        # row `age` is the cumulative distribution function
        # over compressed word indices for an agent of that age in steps
```

`max_life_steps = 3600` (`agent.py:78`) — the model uses 36 steps per simulated year × 100 years.

At each age, the CDF has a different length (an agent at age 1 has access to maybe a few hundred word slots; an adult has access to the full ~1000). The row at index `age` is the lookup table passed to `randZipf(cdf, n)` which does inverse-CDF sampling via binary search.

### 3.2 `lang_spoken_ics_vs_step.h5` — initial-condition memory states

This file is consumed through three-level indexing of the form `self.model.lang_ICs[pct_key][field][age]` (e.g. `agent.py:124, 126, 133`). So the schema is:

```
lang_spoken_ics_vs_step.h5
├── '10_pct'                         # proficiency buckets (ic_pct_keys = [10, 25, 50, 75, 90]
│   ├── 't'                          #  plus an implicit '100_pct' for monolinguals)
│   │   └── <ndarray, shape (max_life_steps, vocab_red)>
│   ├── 'S'
│   │   └── <ndarray, shape (max_life_steps, vocab_red)>
│   └── 'wc'
│       └── <ndarray, shape (max_life_steps, vocab_red)>
├── '25_pct'
│   └── ... (same structure)
├── '50_pct'
├── '75_pct'
├── '90_pct'
└── '100_pct'
```

Where:

- `pct_key` ∈ `{'10_pct', '25_pct', '50_pct', '75_pct', '90_pct', '100_pct'}` — the fraction of an agent's life that this language has occupied (a "lifetime exposure bucket").
- `t` — elapsed steps since last activation per word
- `S` — Ebbinghaus stability per word
- `wc` — encounter count per word

Storage estimate: 6 buckets × 3 fields × 3600 ages × 1000 words × 8 bytes = **~520 MB**. That's why the data ships out-of-band.

Crucially, the ICs are **not at-birth states** — they're snapshots indexed by age. When a 30-year-old agent is born into the simulation, the model reads the row at `age=30` from the bucket matching the agent's lifetime language exposure. This is how the simulator seeds a non-trivial demography in step 0 without having to run a millenium of warmup.

The retrievability array `R` is *not* stored; it's computed on the fly via `R = exp(-k · t / S)` (`agent.py:128-131`).

## 4. Can they be rebuilt?

### 4.1 `lang_cdfs_vs_step.h5` — **yes, trivially**

All the math is already in `zipf_generator/Zipf.py`. The file is just a precomputed lookup table; you can regenerate it with a 20-line script:

```python
# regenerate_cdfs.py (sketch — not in the repo)
import numpy as np, deepdish as dd
from bilangsim.zipf_generator import Zipf_Mand_CDF_compressed

max_life_steps = 3600
full_vocab     = 10_000          # raw vocabulary size before compression
vocab_red      = 1000            # compressed slots

cdfs_by_age = np.empty((max_life_steps, vocab_red))
for age in range(max_life_steps):
    n = vocab_growth_curve(age, full_vocab)   # age-dependent vocab ceiling
    cdfs_by_age[age] = Zipf_Mand_CDF_compressed(n, alpha=1.5, n_red=vocab_red)

dd.io.save('lang_cdfs_vs_step.h5', {'s': cdfs_by_age})
```

The only piece not given in the codebase is the exact `vocab_growth_curve(age)` shape — but the design comment at `Zipf.py:70-71` says:

> *"IDEA : to model vocab_size vs age dependency, play both with n and n_red in following function. Use factor for n, n_red ???"*

…which suggests the original generator parameterised vocabulary growth by simply scaling `n` with age. A sigmoid from a few hundred words (early childhood) to ~10k (adulthood) would match real lexicon-growth literature.

### 4.2 `lang_spoken_ics_vs_step.h5` — **rebuildable in principle, but the recipe isn't in the repo**

No generator script is shipped. Inspection of the consumer code suggests these arrays are the **fixed-point output of a warmup simulation**:

- For each exposure bucket *p* ∈ {10, 25, 50, 75, 90, 100}%, run a virtual agent through a full life trajectory where language X gets *p*% of their conversational exposure.
- Snapshot `(t, S, wc)` at every age step.
- Save.

Implementing this means writing a single-agent warmup driver that uses the same Ebbinghaus update rules (`numba_comp_delta_S` at `agent.py:16` and the decay loop at `agent.py:725-753`). It's ~100 lines of code, but you'd have to be careful that the warmup uses an exposure schedule consistent with what an actual agent in a full simulation encounters — otherwise the ICs don't match the steady-state dynamics, and step-0 agents will exhibit a transient.

Alternative: ship the model with `_set_null_lang_attrs` ICs (`S=0.01, t=1000`, see `agent.py:145-168`) for every agent and add a configurable warmup phase to `run_model`. This trades startup time for simplicity and removes the HDF5 dependency entirely.

## 5. Why HDF5 at all? And why `deepdish`?

`deepdish` (`dd.io.save` / `dd.io.load`) is a thin wrapper that lets you serialise an arbitrary nested Python `dict` of NumPy arrays into a single HDF5 file. The legacy project picked it because the natural data structure is exactly that: nested dicts of arrays. HDF5 gave:

- single-file packaging
- chunked + compressed storage of large dense arrays
- per-array random access (no need to deserialise everything to read one bucket)

The downsides have aged poorly:

- `deepdish` is essentially unmaintained (last release 2020, requires `tables` ≥ 3.4 which itself has stagnated).
- HDF5 has well-known concurrency and file-locking issues, brittle Python wheel installs, and binary-format opacity (the file can't be diffed or grep'd).
- The package pins `Cython==0.28.5`, `numpy==1.15.0`, `pandas==0.23.4`, `tables==3.4.4` — a 2018-era stack that no longer installs cleanly on modern Python.

For a fresh build of this project on Python 3.14 + NumPy 2.x, *something* has to change. Options below.

## 6. Replacement options

The two files have different storage profiles and should probably be handled separately.

### Option A — Drop persistent storage, generate at runtime

Both files are read-only configuration. The CDFs in particular cost milliseconds to compute from the Zipf formulas. Building them on `BiLangModel.__init__` would:

- Eliminate the HDF5 dependency entirely.
- Remove 500+ MB of binary data from version control / distribution.
- Make parameter changes (Zipf exponents, vocab ceilings) trivial — no regenerate-and-recommit cycle.

Downside: agent IC generation takes longer if you also synthesise `lang_ICs` via warmup. But this can be cached on first run to `~/.cache/bilangsim/` and reused. **Recommended for the CDFs unconditionally; for the ICs if startup time is tolerable.**

### Option B — NumPy `.npz` (multi-array zipped archive)

The simplest possible drop-in. `np.savez_compressed("ics.npz", **flat_dict)` and `np.load("ics.npz")`. You'd flatten the bucket dimension into key prefixes (`"10_pct__t"`, `"10_pct__S"`, …).

Pros: zero new dependencies (NumPy ships with it), small files, fast load (mmap-able with `np.load(..., mmap_mode='r')`).
Cons: schema lives in naming conventions rather than the file. Loses the nested-dict feel.

### Option C — Zarr (modern HDF5 successor)

Zarr stores chunked, compressed N-dimensional arrays as a directory of small files (or a single zip file, or cloud object-store keys). It's the de facto NumPy-native columnar/array store in 2025.

Pros:
- Active ecosystem (`zarr-python` v3, NumPy 2.x compatible, NetCDF-compatible via xarray).
- Preserves the nested-group structure (`/10_pct/t`, `/10_pct/S`) almost identically to HDF5.
- Supports cloud-native access (S3, GCS) if the project ever distributes data remotely.
- Chunking + compression are first-class, configurable per array.

Cons: adds one runtime dependency. Directory-of-files layout is mildly inconvenient (can be zipped into a single `.zip` store if you want a single artefact).

**Recommended if you keep persistent storage at all.**

### Option D — xarray + NetCDF (or Zarr)

The dataset is fundamentally a 4-dimensional labelled array: `(bucket, field, age, word_index)`. xarray models this natively. NetCDF is "HDF5 with standardised conventions" and is the default scientific Python lingua franca for indexed multi-dim data.

```python
import xarray as xr
ds = xr.Dataset({
    field: (("bucket", "age", "word"), array)
    for field, array in ...
})
ds.to_netcdf("lang_ics.nc")
# or
ds.to_zarr("lang_ics.zarr")
```

Pros: dimensions and coordinates are named, so the brittle `dict[pct_key][field][age]` access becomes `ds.sel(bucket="10_pct", age=42).t`. Refactor-resilient and self-documenting.

Cons: more conceptual overhead. Adds `xarray` (heavy-ish dep) and `netCDF4` or `zarr` as a backend.

**Best fit if you plan to expose this dataset to other researchers or do exploratory analysis on it.**

### Option E — Apache Parquet (long-format)

Reshape the data into a row-per-cell table: `(bucket, field, age, word_index, value)`. Store as Parquet.

Pros: queryable from DuckDB / Polars / Pandas, excellent compression, columnar.
Cons: ~3600 × 1000 × 6 × 3 ≈ 65M rows. Random access by `(bucket, age)` is fine for analytics but slower than dense-array load for the simulator's "read everything at startup" pattern. **Probably overkill for a simulation init step.**

### Option F — SQLite

Relational store of `(bucket, field, age, word_index) → value`. Fine as a query store, but dense numeric N-D data doesn't belong in a row store. **Not recommended.**

### Option G — DuckDB

Same shape as Parquet (long-format) but with embedded SQL. Useful if you ever want to *analyse* the ICs (e.g. `SELECT age, AVG(S) FROM ics WHERE bucket='50_pct' GROUP BY age`), but for the simulator's read-once-then-index pattern, it adds latency for no benefit.

## 7. Recommendation

For a hypothetical revival of the legacy simulator on the current Python stack:

1. **Delete `data/cdfs/lang_cdfs_vs_step.h5` from the design.** Add `BiLangModel._build_cdfs()` that calls `Zipf_Mand_3S_CDF_comp` for each age step at init. ~15 lines of code, ~10 ms at startup. No binary data needed.

2. **Replace `data/init_conds/lang_spoken_ics_vs_step.h5` with either:**
   - **(a)** A `BiLangModel._warmup()` routine that produces the ICs on first run and caches them to `~/.cache/bilangsim/ics_v1.zarr` keyed by a hash of the relevant parameters. *Best for reproducibility and small dependency surface.*
   - **(b)** A `data/ics.zarr/` directory (or `ics.npz`) shipped alongside the code in the repo. Zarr if you want the named-dimension ergonomics, NPZ if you want zero new dependencies. *Best if you absolutely need bit-identical startup state.*

3. **Drop `deepdish` and `tables`** from `setup.py`. Add `zarr` (or nothing, if you went with NPZ).

4. **If anyone needs the original HDF5s for historical reproduction**, the upstream GitHub repo (`pgervila/bilingual_society_ABM`) is the canonical archive — request them from the author or check for release artefacts.

This trades a couple of hundred MB of opaque binary data for ~150 lines of generator code that is version-controlled, readable, and parameter-tweakable.
