# `zipf_generator/` — word-frequency CDF math

Standalone package providing the Zipf/Zipf-Mandelbrot cumulative distribution
functions the model samples word *intents* from. `Zipf.py` (~126 lines) +
`__init__.py` (exports).

## Exports (`__init__.py`)

```python
from .Zipf import Zipf_CDF, Zipf_Mandelbrot_CDF, randZipf
from .Zipf import Zipf_CDF_compressed, Zipf_Mand_CDF_compressed
from .Zipf import Zipf_Mand_3S_CDF_comp, vocab_ceiling_curve
```

## The CDF variants

| Function | Purpose |
|---|---|
| `Zipf_CDF(n, alpha)` | Pure Zipf CDF over `n` ranks. |
| `Zipf_Mandelbrot_CDF(n, alpha, beta=2.7)` | Zipf-Mandelbrot (adds `beta` offset). |
| `Zipf_Mand_3S_CDF(n, ...)` | **Three-stage** Zipf-Mandelbrot — the realistic one. Three power-law regimes joined at `N1=100` and `N2=2000`, fit to BNC spoken-corpus data. Full length `n`. |
| `Zipf_CDF_compressed` / `Zipf_Mand_CDF_compressed` | Compressed variants: map a long `n`-length CDF onto `n_red` slots by sampling at `linspace(1, n, n_red+1)` interval boundaries. |
| `Zipf_Mand_3S_CDF_comp(n, ..., n_red=1000)` | **The one the model uses.** Compressed three-stage Zipf-Mandelbrot. Called by `BiLangModel._build_cdfs` for every age. |

The three-stage fit constants (`alpha_1=1.16, alpha_2=1.48, alpha_3=1.866,
beta=6.9, c1=2041507.88, c2=9105.72, c3=126.287, N1=100, N2=2000`) come from
curve-fitting the British National Corpus (documented in the recovered research
notebook; see `../../scripts/build_ics.py` provenance notes).

## Compression scheme

Each compressed slot represents a *band* of real-frequency-adjacent words. Slot 0 ≈
the most common word group ("the", "and"), slot `n_red-1` ≈ the rarest sampled. This
is why per-word memory arrays are length `vocab_red` (500/1000), not the raw
vocabulary size.

## `vocab_ceiling_curve(age_steps, steps_per_year=36, n_min=500, n_max=10000, midpoint_years=8, rate=0.4)` (`Zipf.py:93`)

**Added during modernization** to replace the legacy externally-precomputed
age→vocabulary-size mapping. Logistic growth in years from `n_min` (early childhood)
to `n_max` (adulthood), midpoint at 8 years. Returns the raw `n` fed to
`Zipf_Mand_3S_CDF_comp` per age, so younger agents sample from a smaller effective
vocabulary. Tuning these params against lexicon-growth literature is a follow-up task.

## `randZipf(zipf_cum_distr, numSamples)` (`Zipf.py:106`)

Fast sampler: draws `numSamples` uniforms and `np.searchsorted`s them into the CDF,
returning compressed word indices. This is the sole word-sampling primitive used by
`pick_vocab` and `study_vocab`.

`randZipf_dim` (`:113`) is an incomplete higher-dimensional variant — not used.
