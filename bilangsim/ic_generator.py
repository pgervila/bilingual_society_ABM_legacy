"""Generator for the agents' linguistic Initial Conditions (ICs) table.

Faithful re-implementation of the recipe that produced the legacy
`lang_spoken_ics_vs_step.h5` file: `get_lang_knowledge`, `study_memory_fun` and
`words_day_factor` from the deleted notebook `lang_memory_model_research.ipynb`
(recovered from commit c44563d, cells 194/195/203), extended to snapshot the
memory state at every age (commit 18cad3c).

A single solitary speaker is exposed, from birth, to a constant fraction
`pct_hours` of a full-immersion daily token flow, and its per-word memory state
(t, S, wc) is recorded at every step. One trajectory is computed per exposure
bucket. The resulting table is indexed exactly as the legacy model expects:

    lang_ICs['{p}_pct'][field][age, word_idx]     field in {'t', 'S', 'wc'}

Buckets are {10, 25, 50, 75, 90, 100}: monolinguals use '100_pct'; bilinguals
use '{k}_pct' for L1 and '{100-k}_pct' for L2 with k in BiLangModel.ic_pct_keys.

Notes:
    * sampling uses a private RandomState, so building the table never perturbs
      the global `random`/`numpy` RNG state the model relies on
    * the table is cached in-process and on disk (see `load_or_build_lang_ics`);
      a cache miss returns exactly the same values as a cache hit
"""
import hashlib
import os
from pathlib import Path

import numpy as np

# bump to invalidate on-disk caches whenever the recipe changes
GENERATOR_VERSION = 1

DEFAULT_BUCKETS = (10, 25, 50, 75, 90, 100)
DEFAULT_SEED = 42

# memory retrievability constant (same as BaseAgent.k)
K = np.log(10 / 9)
# SM-2 parameters (notebook cell 194; same values as agent.update_words_memory)
A, B, C, D = 7.6, 0.023, -0.031, -0.2
# speech scale factor accounting for own speech, TV, radio (notebook cell 194)
F_S = 2.0
# minimum lifetime count before a word starts being memorised (notebook cell 194)
ENCODING_THRESHOLD = 5
# per-step stability decay after age 65 (notebook cell 194)
POST_65_DECAY = 0.01
# number of days in one step (36 steps per year)
DAYS_PER_STEP = 10

# on-disk storage dtypes; values are upcast to these legacy dtypes when loaded
_DISK_DTYPES = {'t': np.float32, 'S': np.float32, 'wc': np.int32}
_RUN_DTYPES = {'t': np.float64, 'S': np.float64, 'wc': np.int64}

_MEMO = {}


def words_day_factor(age):
    """ Notebook cell 194, verbatim. Coefficient that sets the number of tokens
        per day as a fraction of compressed vocabulary size, assuming ~16000 tokens
        per adult per day. Three-piece curve: high in childhood, plateau at 2.5 in
        adulthood, mild rise in old age. Age is expressed in steps """
    if age < 36 * 14:
        return 2.5 + 100 * np.exp(-0.014 * age)
    if 36 * 14 <= age <= 36 * 65:
        return 2.5
    return 1.5 + np.exp(0.002 * (age - 36 * 65))


def build_bucket_trajectory(pct_hours, cdfs, rng, a=A, b=B, c=C, d=D):
    """ Simulate a solitary speaker exposed to a fraction `pct_hours` of full daily
        token flow, from birth until cdfs.shape[0] steps of age
        Args:
            * pct_hours: float in (0, 1]. Exposure fraction
            * cdfs: numpy array (max_steps, n_red). Age-indexed compressed Zipf CDFs
            * rng: numpy RandomState used for word sampling
            * a, b, c, d: SM-2 memory-model parameters
        Output:
            * t, S, wc: numpy arrays of shape (max_steps, n_red) with the memory
              state snapshot after each step
    """
    max_steps, n_red = cdfs.shape
    S = np.full(n_red, 0.01, dtype=np.float64)
    t = np.full(n_red, 100.0, dtype=np.float64)
    wc = np.zeros(n_red, dtype=np.int64)
    R = np.zeros(n_red, dtype=np.float64)

    t_out = np.empty((max_steps, n_red), dtype=np.float64)
    S_out = np.empty((max_steps, n_red), dtype=np.float64)
    wc_out = np.empty((max_steps, n_red), dtype=np.int64)

    for age in range(max_steps):
        words_per_day = n_red / words_day_factor(age)
        n_samples = int(F_S * pct_hours * words_per_day * DAYS_PER_STEP)
        # same sampling as zipf_generator.randZipf, with a private RNG
        zipf_samples = np.searchsorted(cdfs[age], rng.random_sample(n_samples))
        act, act_c = np.unique(zipf_samples, return_counts=True)
        wc[act] += act_c

        # only words heard more than ENCODING_THRESHOLD times can be memorised
        mem_availab = np.where(wc > ENCODING_THRESHOLD)[0]
        keep = np.isin(act, mem_availab, assume_unique=True)
        act, act_c = act[keep], act_c[keep]

        if act.size:
            # stability increase due to first (re)activation in step
            S[act] += a * (S[act] ** -b) * np.exp(c * 100 * R[act]) + d
            mask = np.zeros(n_red, dtype=bool)
            mask[act] = True
            t[~mask] += 1
            t[mask] = 0
            # remaining activations applied in bulk ("simplification with good approx")
            act_c = act_c - 1
            S[act] += act_c * (a * (S[act] ** -b) * np.exp(c * 100 * R[act]) + d)
        else:
            t += 1

        # memory becomes shakier after turning 65
        if age > 65 * 36:
            S = np.where(S >= 0.01, S - POST_65_DECAY, 0.000001)

        R = np.exp(-K * t / S)

        t_out[age] = t
        S_out[age] = S
        wc_out[age] = wc

    return t_out, S_out, wc_out


def build_lang_ics(cdfs, buckets=DEFAULT_BUCKETS, seed=DEFAULT_SEED):
    """ Build the full IC table (no caching, full float64/int64 precision)
        Args:
            * cdfs: numpy array (max_steps, n_red). Age-indexed compressed Zipf CDFs
            * buckets: iterable of integer exposure percentages
            * seed: integer. Seed of the private RNG
        Output:
            * dict {'{p}_pct': {'t': array, 'S': array, 'wc': array}}
    """
    rng = np.random.RandomState(seed)
    table = {}
    for pct in buckets:
        t, S, wc = build_bucket_trajectory(pct / 100, cdfs, rng)
        table[f'{pct}_pct'] = {'t': t, 'S': S, 'wc': wc}
    return table


def default_cache_dir():
    """ $BILANGSIM_CACHE_DIR, else $XDG_CACHE_HOME/bilangsim, else ~/.cache/bilangsim """
    if os.environ.get('BILANGSIM_CACHE_DIR'):
        return Path(os.environ['BILANGSIM_CACHE_DIR'])
    xdg = os.environ.get('XDG_CACHE_HOME')
    base = Path(xdg) if xdg else Path.home() / '.cache'
    return base / 'bilangsim'


def _cache_key(cdfs, buckets, seed):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(cdfs, dtype=np.float64).tobytes())
    h.update(repr((cdfs.shape, tuple(buckets), seed, GENERATOR_VERSION)).encode())
    return h.hexdigest()[:16]


def _quantize(table):
    """ Round-trip through on-disk dtypes so cache misses and hits are identical """
    return {key: {f: arr.astype(_DISK_DTYPES[f]) for f, arr in store.items()}
            for key, store in table.items()}


def _to_runtime(disk_table):
    """ Upcast to legacy runtime dtypes and freeze arrays (agents copy them) """
    out = {}
    for key, store in disk_table.items():
        out[key] = {}
        for f, arr in store.items():
            run_arr = arr.astype(_RUN_DTYPES[f])
            run_arr.setflags(write=False)
            out[key][f] = run_arr
    return out


def _save(disk_table, path):
    arrays = {f'{key}__{f}': arr for key, store in disk_table.items()
              for f, arr in store.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
    # atomic rename so concurrent runs never read a partial file
    os.replace(tmp, path)


def _load(path):
    disk_table = {}
    with np.load(path) as data:
        for name in data.files:
            key, f = name.split('__')
            disk_table.setdefault(key, {})[f] = data[name]
    return disk_table


def load_or_build_lang_ics(cdfs, buckets=DEFAULT_BUCKETS, seed=DEFAULT_SEED,
                           cache_dir=None, use_disk_cache=True):
    """ Return the IC table, using an in-process memo and an on-disk cache.
        Cache is keyed by the CDFs content, buckets, seed and GENERATOR_VERSION.
        Args:
            * cdfs: numpy array (max_steps, n_red). Age-indexed compressed Zipf CDFs
            * buckets: iterable of integer exposure percentages
            * seed: integer. Seed of the private RNG
            * cache_dir: directory for on-disk cache. Defaults to `default_cache_dir()`
            * use_disk_cache: boolean. If False, only the in-process memo is used
        Output:
            * dict {'{p}_pct': {'t', 'S', 'wc'}} of read-only arrays of shape
              (max_steps, n_red), dtypes float64 / float64 / int64
    """
    key = _cache_key(cdfs, buckets, seed)
    if key in _MEMO:
        return _MEMO[key]

    path = Path(cache_dir or default_cache_dir()) / f'lang_ics_{key}.npz'
    disk_table = None
    if use_disk_cache and path.exists():
        try:
            disk_table = _load(path)
        except (OSError, ValueError, KeyError):
            # corrupt or incompatible cache file: rebuild below
            disk_table = None
    if disk_table is None:
        disk_table = _quantize(build_lang_ics(cdfs, buckets=buckets, seed=seed))
        if use_disk_cache:
            try:
                _save(disk_table, path)
            except OSError:
                # read-only filesystem etc.: proceed without disk cache
                pass

    table = _to_runtime(disk_table)
    _MEMO[key] = table
    return table
