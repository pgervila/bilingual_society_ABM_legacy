import random

import numpy as np
import pytest

from bilangsim import ic_generator as icg
from bilangsim.zipf_generator import Zipf_Mand_3S_CDF_comp, vocab_ceiling_curve

MAX_STEPS = 60
VOCAB_RED = 40


@pytest.fixture(scope='module')
def cdfs():
    """Small age-indexed CDF stack built with the same recipe as BiLangModel._build_cdfs"""
    out = np.empty((MAX_STEPS, VOCAB_RED))
    for age in range(MAX_STEPS):
        n = max(int(vocab_ceiling_curve(age)), VOCAB_RED)
        out[age] = Zipf_Mand_3S_CDF_comp(n, n_red=VOCAB_RED)
    return out


@pytest.fixture(autouse=True)
def clear_memo():
    icg._MEMO.clear()
    yield
    icg._MEMO.clear()


def test_keys_shapes_dtypes(cdfs):
    table = icg.build_lang_ics(cdfs)
    assert set(table) == {'10_pct', '25_pct', '50_pct', '75_pct', '90_pct', '100_pct'}
    for store in table.values():
        assert set(store) == {'t', 'S', 'wc'}
        for field in ('t', 'S', 'wc'):
            assert store[field].shape == (MAX_STEPS, VOCAB_RED)
        assert store['t'].dtype == np.float64
        assert store['S'].dtype == np.float64
        assert store['wc'].dtype == np.int64


def test_bilingual_complement_keys_exist(cdfs):
    """legacy set_lang_ics uses '{k}_pct' and '{100-k}_pct' for every k in ic_pct_keys"""
    from bilangsim.model import BiLangModel
    table = icg.build_lang_ics(cdfs)
    for k in BiLangModel.ic_pct_keys:
        assert f'{k}_pct' in table
        assert f'{100 - k}_pct' in table


def test_deterministic_given_seed(cdfs):
    t1 = icg.build_lang_ics(cdfs, seed=7)
    t2 = icg.build_lang_ics(cdfs, seed=7)
    t3 = icg.build_lang_ics(cdfs, seed=8)
    for key in t1:
        for field in ('t', 'S', 'wc'):
            np.testing.assert_array_equal(t1[key][field], t2[key][field])
    assert not np.array_equal(t1['50_pct']['wc'], t3['50_pct']['wc'])


def test_global_rng_untouched(cdfs):
    np.random.seed(123)
    random.seed(123)
    np_state = np.random.get_state()[1].copy()
    py_state = random.getstate()
    icg.build_lang_ics(cdfs)
    np.testing.assert_array_equal(np.random.get_state()[1], np_state)
    assert random.getstate() == py_state


def test_exposure_ordering(cdfs):
    table = icg.build_lang_ics(cdfs)
    wc_10 = table['10_pct']['wc']
    wc_100 = table['100_pct']['wc']
    # word counts never decrease with age
    assert np.all(np.diff(wc_100, axis=0) >= 0)
    # more exposure -> more cumulative tokens
    assert wc_100[-1].sum() > wc_10[-1].sum()


def test_cache_roundtrip_and_hit_equals_miss(cdfs, tmp_path):
    miss = icg.load_or_build_lang_ics(cdfs, cache_dir=tmp_path)
    assert len(list(tmp_path.glob('lang_ics_*.npz'))) == 1
    icg._MEMO.clear()
    hit = icg.load_or_build_lang_ics(cdfs, cache_dir=tmp_path)
    assert hit is not miss
    for key in miss:
        for field in ('t', 'S', 'wc'):
            np.testing.assert_array_equal(miss[key][field], hit[key][field])
            assert miss[key][field].dtype == hit[key][field].dtype


def test_memo_returns_same_object(cdfs, tmp_path):
    a = icg.load_or_build_lang_ics(cdfs, cache_dir=tmp_path)
    b = icg.load_or_build_lang_ics(cdfs, cache_dir=tmp_path)
    assert a is b


def test_cache_key_depends_on_cdfs_and_seed(cdfs, tmp_path):
    icg.load_or_build_lang_ics(cdfs, cache_dir=tmp_path)
    icg.load_or_build_lang_ics(cdfs, seed=99, cache_dir=tmp_path)
    other_cdfs = np.array([Zipf_Mand_3S_CDF_comp(max(int(vocab_ceiling_curve(age)), 20), n_red=20)
                           for age in range(MAX_STEPS)])
    icg.load_or_build_lang_ics(other_cdfs, cache_dir=tmp_path)
    assert len(list(tmp_path.glob('lang_ics_*.npz'))) == 3


def test_tables_are_read_only(cdfs, tmp_path):
    table = icg.load_or_build_lang_ics(cdfs, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        table['100_pct']['S'][0, 0] = 1.0
