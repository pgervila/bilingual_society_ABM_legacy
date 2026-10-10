"""Build (or load from cache) the agents' linguistic IC table and print sanity checks.

Thin CLI over `bilangsim.ic_generator`. Running it warms the same on-disk cache
`BiLangModel` uses at construction (same CDFs, buckets and seed => same cache key),
so the first model construction afterwards loads instead of generating.

Run:
    PYTHONHASHSEED=0 uv run python scripts/build_ics.py [--vocab-red 500] [--seed 42] [--no-cache]
"""
import argparse
import time

import numpy as np

from bilangsim import ic_generator as icg
from bilangsim.model import BiLangModel
from bilangsim.zipf_generator import age_indexed_cdfs


def report(table, vocab_red):
    age_5, age_30, age_70 = 36 * 5, 36 * 30, 36 * 70
    store = table['100_pct']
    print("\n=== Bucket 100_pct (full exposure) ===")
    for label, age in (('5y', age_5), ('30y', age_30)):
        S, wc = store['S'][age], store['wc'][age]
        print(f"  age={label:>3}: S[0:5] = {np.round(S[:5], 3)}")
        print(f"            S[-5:] = {np.round(S[-5:], 3)}")
        print(f"            encoded (wc>{icg.ENCODING_THRESHOLD}) = "
              f"{(wc > icg.ENCODING_THRESHOLD).sum():>4d}/{vocab_red}")

    print("\n=== Per bucket: cumulative tokens at 30y, mean R and pct(R>0.9) ===")
    print(f"  {'bucket':>8} | {'tokens@30y':>11} | {'R@5y':>5} | {'R@30y':>5} | {'R@70y':>5} "
          f"| {'pct@5y':>6} | {'pct@30y':>7}")
    for key in sorted(table, key=lambda k: int(k.split('_')[0])):
        st = table[key]
        R = {a: np.exp(-icg.K * st['t'][a] / st['S'][a]) for a in (age_5, age_30, age_70)}
        print(f"  {key:>8} | {st['wc'][age_30].sum():>11,} | {R[age_5].mean():>5.3f} | "
              f"{R[age_30].mean():>5.3f} | {R[age_70].mean():>5.3f} | "
              f"{(R[age_5] > 0.9).mean():>6.3f} | {(R[age_30] > 0.9).mean():>7.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--vocab-red', type=int, default=500,
                        help='Compressed vocabulary size (500 = spoken-only model default).')
    parser.add_argument('--seed', type=int, default=icg.DEFAULT_SEED)
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not read or write the on-disk cache.')
    args = parser.parse_args()

    cdfs = age_indexed_cdfs(args.vocab_red, BiLangModel.max_life_steps,
                            steps_per_year=BiLangModel.steps_per_year)
    t0 = time.time()
    table = icg.load_or_build_lang_ics(cdfs, seed=args.seed,
                                       use_disk_cache=not args.no_cache)
    print(f"IC table ready in {time.time() - t0:.1f}s "
          f"(buckets={icg.DEFAULT_BUCKETS}, vocab_red={args.vocab_red}, "
          f"cache={'off' if args.no_cache else icg.default_cache_dir()})")
    report(table, args.vocab_red)


if __name__ == '__main__':
    main()
