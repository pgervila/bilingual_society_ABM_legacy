"""Reproduce the legacy IC-table generator.

Faithful Python re-implementation of `get_lang_knowledge`, `study_memory_fun`
and `words_day_factor` from the deleted notebook `lang_memory_model_research.ipynb`
(recovered from commit c44563d, cells 194/195/203). Produces the same
`lang_ICs[pct_key][field][age, word_idx]` table the 2019 model loaded from
`lang_spoken_ics_vs_step.h5` -- spoken-only path (read_hours_per_day=0), the
variant the legacy code actually used in `BaseAgent._set_lang_attrs`.

Run:
    PYTHONHASHSEED=0 uv run python scripts/build_ics.py

Prints the four sanity values from the agreed design pass:
  * lang_ICs[100]['S'][age=36*5]   (5y, 100%): top words stable, tail near floor
  * lang_ICs[100]['S'][age=36*30]  (30y, 100%): broad shoulder
  * lang_ICs[25]['wc'][age=36*30] vs 100%: should be ~1/4 the tokens
  * mean(R(age)) per bucket: rises in childhood, plateaus, softens after 65
"""
from __future__ import annotations

import random
import time

import numpy as np

from bilangsim.zipf_generator import (
    Zipf_Mand_3S_CDF_comp,
    randZipf,
    vocab_ceiling_curve,
)


# Memory retrievability constant (legacy: k = ln(10/9))
K = np.log(10 / 9)

# Match the live model's constants so the table can be plugged in directly.
VOCAB_RED = 500
MAX_LIFE_STEPS = 3600
STEPS_PER_YEAR = 36

# Notebook (cell 203) used {25, 50, 75, 100}. Sticking to that for the first
# reproduction; widening to {10, 25, 50, 75, 90, 100} is a separate decision.
BUCKETS = (25, 50, 75, 100)

# SM-2 parameters (notebook cell 194). Identical to numba_comp_delta_S in agent.py.
A, B, C, D = 7.6, 0.023, -0.031, -0.2

# Notebook constants. f_s is the "speech vocab" scale factor accounting for
# self-talk / TV / radio on top of incoming conversation tokens.
F_S = 2.0
ENCODING_THRESHOLD = 5  # word_counter > 5 to start memorising (cell 194)
POST_65_DECAY = 0.01    # per-step stability decay after age 65 (cell 194)

SEED = 42


def words_day_factor(age: int) -> float:
    """Notebook cell 194, verbatim.

    Coeff that determines num hours spoken per day as pct of vocabulary size,
    assuming ~16000 tokens per adult per day as average. Three-piece curve:
    childhood high, adult plateau 2.5, elderly mild rise.
    """
    if age < 36 * 14:
        return 2.5 + 100 * np.exp(-0.014 * age)
    if 36 * 14 <= age <= 36 * 65:
        return 2.5
    return 1.5 + np.exp(0.002 * (age - 36 * 65))


def build_cdfs(n_red: int = VOCAB_RED,
               max_steps: int = MAX_LIFE_STEPS,
               steps_per_year: int = STEPS_PER_YEAR) -> np.ndarray:
    """Same recipe as `BiLangModel._build_cdfs` -- reproduce it standalone so
    this script does not need to construct a full model."""
    cdfs = np.empty((max_steps, n_red), dtype=np.float64)
    for age in range(max_steps):
        n = max(int(vocab_ceiling_curve(age, steps_per_year=steps_per_year)), n_red)
        cdfs[age] = Zipf_Mand_3S_CDF_comp(n, n_red=n_red)
    return cdfs


def build_bucket_trajectory(pct_hours: float,
                            cdfs: np.ndarray,
                            n_red: int = VOCAB_RED,
                            max_steps: int = MAX_LIFE_STEPS,
                            a: float = A, b: float = B,
                            c: float = C, d: float = D):
    """Reproduce `get_lang_knowledge` and snapshot (t, S, wc) per step.

    Returns three arrays of shape (max_steps, n_red): `t_out, S_out, wc_out`.
    The per-step snapshot was the extension introduced in commit 18cad3c
    (`Enabled age-specific import of t,S,wc Initial Conditions`) -- the
    original notebook returned only the final state.
    """
    S = np.full(n_red, 0.01, dtype=np.float64)
    t = np.full(n_red, 100.0, dtype=np.float64)
    wc = np.zeros(n_red, dtype=np.int64)
    R = np.zeros(n_red, dtype=np.float64)

    t_out = np.empty((max_steps, n_red), dtype=np.float64)
    S_out = np.empty((max_steps, n_red), dtype=np.float64)
    wc_out = np.empty((max_steps, n_red), dtype=np.int64)

    for age in range(max_steps):
        words_per_day = n_red / words_day_factor(age)
        n_samples = int(F_S * pct_hours * words_per_day * 10)
        zipf_samples = randZipf(cdfs[age], n_samples)
        act, act_c = np.unique(zipf_samples, return_counts=True)

        wc[act] += act_c

        # Encoding gate -- only words seen more than ENCODING_THRESHOLD times
        # contribute to SM-2 updates.
        mem_availab = np.where(wc > ENCODING_THRESHOLD)[0]
        keep = np.isin(act, mem_availab, assume_unique=True)
        act = act[keep]
        act_c = act_c[keep]

        if act.size:
            # SM-2 stability update (notebook math, identical to numba_comp_delta_S)
            delta_S = a * (S[act] ** -b) * np.exp(c * 100 * R[act]) + d
            S[act] += delta_S

            mask = np.zeros(n_red, dtype=bool)
            mask[act] = True
            t[~mask] += 1
            t[mask] = 0

            # Apply remaining `act_c - 1` activations in bulk using updated S.
            # (The notebook calls this a "simplification with good approx".)
            act_c -= 1
            delta_S2 = act_c * (a * (S[act] ** -b) * np.exp(c * 100 * R[act]) + d)
            S[act] += delta_S2
        else:
            # No words crossed the encoding gate this step; all t age by 1.
            t += 1

        # Post-65 stability decay (notebook cell 194)
        if age > 65 * 36:
            S = np.where(S >= 0.01, S - POST_65_DECAY, 0.000001)

        R = np.exp(-K * t / S)

        t_out[age] = t
        S_out[age] = S
        wc_out[age] = wc

    return t_out, S_out, wc_out


def build_lang_ics(buckets=BUCKETS, n_red=VOCAB_RED, max_steps=MAX_LIFE_STEPS, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    cdfs = build_cdfs(n_red=n_red, max_steps=max_steps)
    table = {}
    for pct in buckets:
        t_arr, S_arr, wc_arr = build_bucket_trajectory(
            pct / 100.0, cdfs, n_red=n_red, max_steps=max_steps,
        )
        table[f"{pct}_pct"] = {"t": t_arr, "S": S_arr, "wc": wc_arr}
    return table, cdfs


def _report(table: dict) -> None:
    age_5 = 36 * 5
    age_30 = 36 * 30
    age_70 = 36 * 70

    print("\n=== Bucket 100_pct (full L1 exposure) ===")
    s5 = table["100_pct"]["S"][age_5]
    s30 = table["100_pct"]["S"][age_30]
    wc5 = table["100_pct"]["wc"][age_5]
    wc30 = table["100_pct"]["wc"][age_30]
    print(f"  age=5y:  S[0:5]   = {np.round(s5[:5], 3)}")
    print(f"           S[-5:]   = {np.round(s5[-5:], 3)}")
    print(f"           wc>0     = {(wc5 > 0).sum():>3d}/{VOCAB_RED}")
    print(f"           wc>{ENCODING_THRESHOLD} (encoded) = {(wc5 > ENCODING_THRESHOLD).sum():>3d}/{VOCAB_RED}")
    print(f"  age=30y: S[0:5]   = {np.round(s30[:5], 3)}")
    print(f"           S[-5:]   = {np.round(s30[-5:], 3)}")
    print(f"           wc>0     = {(wc30 > 0).sum():>3d}/{VOCAB_RED}")
    print(f"           wc>{ENCODING_THRESHOLD}        = {(wc30 > ENCODING_THRESHOLD).sum():>3d}/{VOCAB_RED}")

    print("\n=== Bucket vs bucket cumulative wc at age=30y ===")
    for pct in (25, 50, 75, 100):
        s = table[f"{pct}_pct"]["wc"][age_30].sum()
        print(f"  {pct:3d}%: total tokens = {s:>10,}")

    print("\n=== Mean R(age) per bucket (snapshot rows) ===")
    print(f"  {'bucket':>8} | {'R@5y':>6} | {'R@30y':>6} | {'R@70y':>6}")
    for pct in (25, 50, 75, 100):
        store = table[f"{pct}_pct"]
        R5 = np.exp(-K * store["t"][age_5] / store["S"][age_5]).mean()
        R30 = np.exp(-K * store["t"][age_30] / store["S"][age_30]).mean()
        R70 = np.exp(-K * store["t"][age_70] / store["S"][age_70]).mean()
        print(f"  {pct:>6d}%  | {R5:>6.3f} | {R30:>6.3f} | {R70:>6.3f}")


if __name__ == "__main__":
    t0 = time.time()
    table, _cdfs = build_lang_ics()
    elapsed = time.time() - t0
    print(f"Built IC table in {elapsed:.1f}s "
          f"(buckets={BUCKETS}, n_red={VOCAB_RED}, max_steps={MAX_LIFE_STEPS}).")
    _report(table)
