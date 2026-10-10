"""Agents' linguistic Initial Conditions come from the per-age IC table (legacy behaviour)"""
import numpy as np
import pytest

from bilangsim import BiLangModel
from bilangsim.agent import Adult


@pytest.fixture(scope='module')
def model():
    return BiLangModel(120, num_clusters=1)


def find_agent(model, language):
    return next(ag for ag in model.schedule.agents
                if ag.info['language'] == language and ag.info['age'] > 0)


def assert_store_from_table(ag, lang, pct_key):
    row = ag.info['age']
    table = ag.model.lang_ICs[pct_key]
    np.testing.assert_array_equal(ag.lang_stats[lang]['t'], table['t'][row])
    np.testing.assert_array_equal(ag.lang_stats[lang]['S'], table['S'][row])
    np.testing.assert_array_equal(ag.lang_stats[lang]['wc'], table['wc'][row])
    np.testing.assert_allclose(ag.lang_stats[lang]['R'],
                               np.exp(-ag.k * table['t'][row] / table['S'][row]))


def assert_store_null(ag, lang):
    assert np.all(ag.lang_stats[lang]['S'] == 0.01)
    assert np.all(ag.lang_stats[lang]['t'] == 1000)
    assert not ag.lang_stats[lang]['wc'].any()


def test_model_has_ic_table(model):
    assert set(model.lang_ICs) == {f'{p}_pct' for p in (10, 25, 50, 75, 90, 100)}
    for store in model.lang_ICs.values():
        for field in ('t', 'S', 'wc'):
            assert store[field].shape == (model.max_life_steps, model.vocab_red)


@pytest.mark.parametrize('language, known, unknown', [(0, 'L1', 'L2'), (2, 'L2', 'L1')])
def test_monolingual_ics(model, language, known, unknown):
    ag = find_agent(model, language)
    ag.set_lang_ics()
    assert_store_from_table(ag, known, '100_pct')
    assert_store_null(ag, unknown)
    assert_store_null(ag, 'L12')
    assert_store_null(ag, 'L21')


@pytest.mark.parametrize('biling_key', [10, 25, 50, 75, 90])
def test_bilingual_ics(model, biling_key):
    ag = find_agent(model, 1)
    ag.set_lang_ics(biling_key=biling_key)
    assert_store_from_table(ag, 'L1', f'{biling_key}_pct')
    assert_store_from_table(ag, 'L2', f'{100 - biling_key}_pct')
    assert_store_null(ag, 'L12')
    assert_store_null(ag, 'L21')


def test_agent_arrays_are_private_writable_copies(model):
    ag = find_agent(model, 0)
    ag.set_lang_ics()
    for field in ('t', 'S', 'wc'):
        arr = ag.lang_stats['L1'][field]
        assert arr.flags.writeable
        assert not np.shares_memory(arr, model.lang_ICs['100_pct'][field])


def test_pct_knowledge_set_at_current_age(model):
    ag = find_agent(model, 0)
    ag.set_lang_ics()
    age = ag.info['age']
    expected = (ag.lang_stats['L1']['R'] > 0.9).mean()
    assert ag.lang_stats['L1']['pct'][age] == pytest.approx(expected)


def test_step0_adults_know_their_language(model):
    """ICs are applied at construction: no warmup needed for non-null knowledge"""
    adults_l1 = [ag for ag in model.schedule.agents
                 if isinstance(ag, Adult) and ag.info['language'] == 0]
    assert adults_l1
    assert np.mean([ag.get_langs_pcts(0) for ag in adults_l1]) > 0.5
