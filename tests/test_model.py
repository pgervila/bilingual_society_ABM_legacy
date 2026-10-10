import sys
import gc
import numpy as np
import pytest
from bilangsim.agent import Adolescent, Young, Adult, Teacher
from bilangsim import BiLangModel


@pytest.fixture(scope="module")
def model():
    return BiLangModel(300, num_clusters=1, init_lang_distrib=[0.25, 0.5, 0.25],
                       immigration=True, pct_immigration=0.005)


@pytest.fixture(scope="function")
def family_ags(model):
    for ag in model.schedule.agents:
        if type(ag) == Adult and ag.info['married']:
            child1, child2 = ag.get_family_relative('child')
            if type(child1) == Adolescent or type(child2) == Adolescent:
                ag1 = ag
                break
    ag2 = ag1.get_family_relative('consort')
    ag3 = [ag for ag in ag.get_family_relative('child') if type(ag) == Adolescent][0]

    return ag1, ag2, ag3


@pytest.fixture(scope="function")
def stranger_ags(model):
    ags = []
    known_ags = set()
    for ix, ag in enumerate(model.schedule.agents):
        if type(ag) == Adult:
            ags.append(ag)
            known_ags.update(set(model.nws.known_people_network[ag]))
            break
    for ag in model.schedule.agents[ix + 1:]:
        if type(ag) == Adult and ag not in known_ags:
            ags.append(ag)
            known_ags.update(set(model.nws.known_people_network[ag]))
        if len(ags) == 4:
            break
    return ags


@pytest.fixture(scope="function")
def family_and_stranger_ags(model):
    class Found(Exception):
        pass

    try:
        for ag in model.schedule.agents:
            if type(ag) == Adult and ag.info['language'] == 1:
                pct1 = ag.lang_stats['L1']['pct'][ag.info['age']]
                pct2 = ag.lang_stats['L2']['pct'][ag.info['age']]
                if pct2 > pct1 > 0.5:
                    for ch in ag.get_family_relative('child'):
                        if type(ch) == Adolescent and ch.lang_stats['L1']['pct'][ch.info['age']] >= 0.3:
                            if model.nws.family_network[ag][ch]['lang'] == 1:
                                raise Found
    except Found:
        parent, child = ag, ch
        try:
            for stranger in model.schedule.agents:
                if type(stranger) == Adult and stranger.info['language'] == 0:
                    if stranger.lang_stats['L2']['pct'][stranger.info['age']] <= 0.05:
                        raise Found
        except Found:
            return parent, child, stranger
    else:
        # TODO: if agents are not found, they should be created to ensure test passes
        return


@pytest.fixture(scope="module", params=[(457, 3), (521, 2), (897, 5), (1501, 6)])
def model_param(request):
    return BiLangModel(request.param[0], num_clusters=request.param[1],
                       init_lang_distrib=[0.25, 0.5, 0.25])

# define cases to test => Inputs and optionally expected results


test_data_comp_cl_centers = [(10, 0.2), (20, 0.1), (26, 0.2)]

test_data_get_conv_params_strangers = [
            ([1, 1], [0.4, 0.4], [0.5, 0.5], ((1, 1), None)),
            ([0, 2], [0.5, 0.04], [0.04, 0.5], ((0, 1), None)),
            ([0, 2], [0.5, 0.04], [0.02, 0.5], ((0, 0), 2)),
            ([1, 1, 1], [0.5, 0.5, 0.4], [0.4, 0.5, 0.5], ((0, 0, 0), None)),
            ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4], ((1, 1, 1), None)),
            ([0, 2, 2], [0.5, 0.02, 0.02], [0.02, 0.5, 0.5], ((0, 0, 0), 2)),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.5, 0.5], ((0, 0, 1), None)),
            ([0, 1, 2], [0.5, 0.5, 0.02], [0.02, 0.4, 0.5], ((0, 0, 0), 2)),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.4, 0.5], ((0, 0, 1), None)),
            ([2, 1, 0], [0.02, 0.5, 0.5], [0.5, 0.4, 0.04], ((1, 1, 1), 0)),
            ([2, 1, 0, 0], [0.04, 0.4, 0.5, 0.5], [0.5, 0.5, 0.02, 0.02], ((1, 1, 1, 1), 0)),
            ([0, 1, 2, 2], [0.5, 0.5, 0.04, 0.02], [0.04, 0.4, 0.5, 0.5], ((0, 0, 0, 0), 2))
]

test_data_get_conv_params_family = [([0, 0], [0.5, 0.5], [0.0, 0.0], {(0, 1): 0}, ((0, 0), None)),
                                    ([1, 1], [0.5, 0.4], [0.4, 0.5], {(0, 1): 1}, ((1, 1), None)),
                                    ([1, 1], [0.5, 0.4], [0.4, 0.5], {(0, 1): 0}, ((0, 0), None)),
                                    ([0, 0, 1], [0.5, 0.5, 0.5], [0.02, 0.04, 0.4],
                                     {(0, 1): 0, (0, 2): 0, (1, 2): 0}, ((0, 0, 0), None)),
                                    ([1, 1, 2], [0.5, 0.5, 0.04], [0.4, 0.4, 0.5],
                                     {(0, 1): 0, (0, 2): 1, (1, 2): 1}, ((1, 1, 1), None)),
                                    ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4],
                                     {(0, 1): 1, (0, 2): 0, (1, 2): 0}, ((0, 0, 0), None))
                                    ]

test_data_run_conversation = [([1, 1, 1, 2, 2], True)]

test_data_remove_after_death = [Adult, Adolescent, Teacher, Young]

test_data_remove_from_locations = [False, True]


#@pytest.mark.skip(reason="too long and currently disabled")
def test_model_consistency(model_param):
    """ Check that all agents live in the same
        cluster as that of their occupation """
    for ag in model_param.schedule.agents:
        if ag.info['age'] > model_param.steps_per_year:
            if isinstance(ag, Young):
                if ag.loc_info['job']:
                    if isinstance(ag, Teacher):
                        assert ag['clust'] == ag.loc_info['job'][0].info['clust']
                    else:
                        assert ag['clust'] == ag.loc_info['job'].info['clust']
            else:
                if ag.loc_info['school']:
                    assert ag['clust'] == ag.loc_info['school'][0].info['clust']

            # assert blocked property is non existent
            assert not hasattr(ag, 'blocked')


def test_run_conversation(model):
    kn_p_nw = model.nws.known_people_network
    for ag in model.schedule.agents:
        if ag.info['language'] == 0 and isinstance(ag, Young):
            ag_init = ag
            break
    for ag in model.schedule.agents:
        if ag.info['language'] == 1 and ag not in kn_p_nw[ag_init]:
            ag1 = ag
            break
    for ag in model.schedule.agents:
        if ag.info['language'] == 2 and ag not in kn_p_nw[ag_init]:
            ag2 = ag
            break
    others = [ag1, ag2]
    # check method does not break if others is empty list
    wc_before = ag_init.lang_stats['L1']['wc']
    model.run_conversation(ag_init, others=[])
    wc_after = ag_init.lang_stats['L1']['wc']
    assert np.all(wc_before == wc_after)
    # check method
    model.run_conversation(ag_init, others)
    # check acquaintance network gets updated by communicating agents
    assert kn_p_nw[ag_init][others[0]]
    assert kn_p_nw[others[0]][ag_init]
    # check agents that can't communicate do not enter acquaintance network
    assert others[1] not in kn_p_nw[ag_init]
    assert ag_init not in kn_p_nw[others[1]]


@pytest.mark.parametrize("langs, pcts_1, pcts_2, expected",
                         test_data_get_conv_params_strangers)
def test_get_conv_params_strangers(model, stranger_ags, langs, pcts_1, pcts_2, expected):
    agents = stranger_ags[:len(langs)]
    for idx, agent in enumerate(agents):
        agent.info['language'] = langs[idx]
        agent.lang_stats['L1']['pct'][agent.info['age']] = pcts_1[idx]
        agent.lang_stats['L2']['pct'][agent.info['age']] = pcts_2[idx]
    params = model.get_conv_params(agents)
    assert np.all(expected == (params['lang_group'], params['mute_type']))


@pytest.mark.parametrize("langs, pcts_1, pcts_2, langs_with_known, expected",
                         test_data_get_conv_params_family)
def test_get_conv_params_family(model, family_ags,
                                langs, pcts_1, pcts_2, langs_with_known, expected):
    agents = family_ags[:len(langs)]
    for idx, agent in enumerate(agents):
        agent.info['language'] = langs[idx]
        agent.lang_stats['L1']['pct'][agent.info['age']] = pcts_1[idx]
        agent.lang_stats['L2']['pct'][agent.info['age']] = pcts_2[idx]
    # set spoken language between acquainted agents
    for ixs, lang in langs_with_known.items():
        model.nws.known_people_network[agents[ixs[0]]][agents[ixs[1]]]['lang'] = lang


def test_get_conv_relatives_with_stranger(model, family_and_stranger_ags):
    if family_and_stranger_ags:
        parent, child, stranger = family_and_stranger_ags
        params = model.get_conv_params([parent, child, stranger])
        assert params['lang_group'] == (0, 0, 0)
        params = model.get_conv_params([parent, child])
        assert params['lang_group'] == (1, 1)


@pytest.mark.parametrize("agent_type", test_data_remove_after_death)
def test_remove_after_death(model, agent_type):
    for ag in model.schedule.agents:
        if type(ag) == agent_type:
            break
    model.remove_after_death(ag)
    gc.collect()
    assert sys.getrefcount(ag) == 2
    assert ag not in model.schedule.agents
    assert ag not in model.nws.known_people_network
    assert ag not in model.nws.family_network
    assert ag not in model.nws.friendship_network


def test_run(model):
    model.run_model(2)


def test_add_immigration_family(model, lang=0, clust=0):
    num_init_agents = model.schedule.get_agent_count()
    model.add_immigration_family(lang, clust)
    num_agents = model.schedule.get_agent_count()
    family = model.schedule.agents[-4:]
    father, mother = family[:2]
    home = father.loc_info['home']
    father_job = father.get_current_job()
    children = family[2:]
    for ag in family:
        assert ag in model.nws.known_people_network
        assert ag in model.nws.friendship_network
        assert ag in model.nws.family_network
        assert ag in home.info['occupants']
        assert ag in model.geo.clusters_info[clust]['agents']

    assert num_agents == num_init_agents + 4
    assert father_job
    assert mother.get_family_relative('child') == children
    for child in children:
        school, course_key = child.get_school_and_course()
        assert child.get_family_relative('father') == father
        assert child.get_family_relative('mother') == mother
        assert child in school.info['students']
        assert child in school[course_key]['students']


def test_add_immigration(model):
    model.add_immigration()



def test_update_frienship(model):
    model.update_friendships()


@pytest.mark.parametrize("replace", test_data_remove_from_locations)
def test_remove_from_locations(model, replace):
    corr_dict = {'Baby': 'Child', 'Child': 'Adolescent', 'Adolescent': 'Young',
                 'YoungUniv': 'Adult', 'Adult': 'Pensioner'}
    # for ag in model.schedule.agents:
    #     if isinstance(ag, corr_dict[agent_type]):
    #         pass # TODO