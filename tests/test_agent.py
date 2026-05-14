import pytest
from unittest.mock import patch

import random
import sys
import gc
import numpy as np
from scipy.spatial.distance import pdist

from bilangsim.agent import Baby, Child, Adolescent, Young, YoungUniv
from bilangsim.agent import Adult, Teacher, TeacherUniv, Pensioner
from bilangsim import BiLangModel


np_seed = np.random.randint(10000)
#np_seed = 8638
np.random.seed(np_seed)
print('test seed is {}'.format(np_seed))


@pytest.fixture(scope="module")
def model():
    return BiLangModel(251, num_clusters=2, check_setup=True)


@pytest.fixture(scope='module')
def dummy_agent(model):
    # find agent that speaks L1
    for ag in model.schedule.agents:
        if ag.info['language'] == 0:
            agent = ag
            break
    return agent


@pytest.fixture(scope='module')
def dummy_baby(model):
    for _ in range(10):
        parent_1 = random.choice([ag for ag in model.schedule.agents if type(ag) == Adult])
        if parent_1.info['married']:
            break
    id_baby = model.set_available_ids.pop()
    # get consort
    consort = parent_1.get_family_relative('consort')
    # find out baby language attribute and langs with parents
    newborn_lang, lang_with_father, lang_with_mother = model.get_newborn_lang(parent_1, consort)
    # Determine baby's sex
    sex = 'M' if random.random() > 0.5 else 'F'
    father, mother = (parent_1, consort) if parent_1.info['sex'] == 'M' else (consort, parent_1)
    # Create baby instance
    baby = Baby(father, mother, lang_with_father, lang_with_mother,
                model, id_baby, newborn_lang, sex, age=40, home=parent_1.loc_info['home'])
    # Add agent to grid, schedule, network and clusters info
    model.add_new_agent_to_model(baby)
    return baby


@pytest.fixture(scope="module")
def city_places(model):
    h1 = model.geo.clusters_info[0]['homes'][0]
    sc1 = model.geo.clusters_info[0]['schools'][0]
    un1 = model.geo.clusters_info[np.argmax([model.geo.cluster_sizes])]['university']
    j1 = model.geo.clusters_info[0]['jobs'][0]
    
    places = {'home': h1, 'school': sc1, 'university': un1, 'job': j1}
    return places


@pytest.fixture(scope="module")
def jobs_in_diff_clusts(model):
    h1 = model.geo.clusters_info[0]['homes'][1]
    h2 = model.geo.clusters_info[1]['homes'][1]
    j1 = model.geo.clusters_info[0]['jobs'][0]
    j2 = model.geo.clusters_info[1]['jobs'][0]


test_data_evolve = [(Baby, Child, ('school', 'school', False)),
                    (Child, Adolescent, ('school', 'school', False)),
                    (Adolescent, Young, ('school', 'job', False)), 
                    (Adolescent, YoungUniv, ('school', 'university', True)),
                    (Young, Adult, ('job', 'job', False)),
                    (YoungUniv, Young, ('university', 'job', False)),
                    (Adult, Pensioner, ('job', None, False)),
                    (Adult, Teacher, ('job', 'job', False)),
                    (Teacher, Pensioner, ('job', None, False)),
                    (Adult, TeacherUniv, ('job', 'job', False))]


test_data_update_lang_arrays = [({'L1': (np.array([0, 1, 10, 450, 470, 490]),
                                         np.array([3, 2, 1, 2, 1, 3]))}, 'listen'),
                                ({'L1': (np.array([0, 1, 5, 10]),
                                         np.array([3, 2, 1, 1]))}, 'speak'),
                                ({'L1': (np.array([450, 470, 490]),
                                         np.array([3, 2, 1]))}, 'listen')]

test_data_vocab_choice_model = [True, False]

test_data_move_new_home = [(True, None, None), (True, True, None), (True, True, True)]

test_data_get_job = [Young, Teacher, TeacherUniv]


def test_pensioner_methods(model):
    for h in model.geo.clusters_info[0]['homes']:
        if not h.info['occupants']:
            home = h
    pensioner_m = Pensioner(model, model.set_available_ids.pop(), 0, 'M',
                            age=36*66, home=h)
    pensioner_f = Pensioner(model, model.set_available_ids.pop(), 0, 'F',
                            age=36*66, home=h)
    model.add_new_agent_to_model(pensioner_m)
    model.add_new_agent_to_model(pensioner_f)

    model.nws.set_link_with_relatives(pensioner_m, pensioner_f, 'consort')
    # pensioner_m.update_acquaintances(pensioner_f, 0)
    # pensioner_m.get_married(pensioner_f)

    child = model.schedule.agents[0]
    grandchild = model.schedule.agents[2]
    model.nws.set_link_with_relatives(child, pensioner_m, 'father')
    model.nws.set_link_with_relatives(grandchild, pensioner_m, 'grandfather')
    pensioner_m.gather_family(freq=1)

    pass


def test_teacher_methods(model, city_places):
    ag_id = model.set_available_ids.pop()
    teacher = Teacher(model, ag_id, 0, 'M', age=1200, home=city_places['home'])
    # test get_current_job
    teacher.loc_info['job'] = None
    teacher.get_current_job()


def test_teacheruniv_methods(model, city_places):
    ag_id = model.set_available_ids.pop()
    teacheruniv = TeacherUniv(model, ag_id, 0, 'M', age=1200, home=city_places['home'])
    model.add_new_agent_to_model(teacheruniv)
    # test get_current_job
    teacheruniv.loc_info['job'] = None
    teacheruniv.get_current_job()


def test_Baby_methods(model, dummy_baby):
    # check creation of needed course in school
    dummy_baby.register_to_school(max_num_studs_per_course=1)
    school_1 = dummy_baby.loc_info['school'][0]
    assert school_1
    assert dummy_baby.loc_info['school'][1]
    # check creation of new school
    # school_1.remove_student(dummy_baby)
    # dummy_baby.register_to_school(max_num_studs_per_course=0)
    # school_2 = dummy_baby.loc_info['school'][0]
    # assert school_1 != school_2
    # assert dummy_baby.loc_info['school'][0]
    # assert dummy_baby.loc_info['school'][1]


def test_Child_methods(model):
    for ag in model.schedule.agents:
        if type(ag) == Child:
            child = ag
            break

    ck = int(child.info['age'] / model.steps_per_year)
    clust_schools = model.geo.clusters_info[child['clust']]['schools']
    if len(clust_schools) == 1:
        model.geo.add_new_school(child['clust'])
        clust_schools = model.geo.clusters_info[child['clust']]['schools']
    clust_schools = sorted(clust_schools, key=lambda x: pdist([x.pos, child.loc_info['home'].pos]))
    old_num_schools = len(clust_schools)

    schools_with_ck = [sc for sc in clust_schools if ck in sc.grouped_studs]
    schools_without_ck = [sc for sc in clust_schools if ck not in sc.grouped_studs]

    child.register_to_school()
    child_school = child.loc_info['school'][0]
    assert child_school
    assert child.loc_info['school'][1]

    for sc in clust_schools:
        if ck in sc.grouped_studs and len(sc.grouped_studs[ck]) <= 25:
            assert child_school in schools_with_ck
            break
    else:
        if schools_without_ck:
            assert child_school in schools_without_ck
        else:
            assert len(clust_schools) == old_num_schools + 1






def test_Adolescent_methods(model):
    pass


def test_Young_methods(model):
    pass


@pytest.mark.parametrize("origin_class, new_class, labels", test_data_evolve)
def test_evolve(model, city_places, origin_class, new_class, labels):
    """ Check correct evolution of one agent class into another """
    if origin_class == Baby:
        father, mother = None, None
        for ag in model.schedule.agents:
            if isinstance(ag, Young) and ag.info['sex'] == 'M':
                father = ag
            elif isinstance(ag, Young) and ag.info['sex'] == 'F':
                mother = ag
            if father and mother:
                break
        ag_id = model.set_available_ids.pop()
        old_ag = origin_class(father, mother, 0, 0, model, ag_id, 0, 'M', age=40,
                              home=city_places['home'],
                              **{labels[0]: city_places[labels[0]]})
        model.add_new_agent_to_model(old_ag)
    elif origin_class == Teacher:
        ag_id = model.set_available_ids.pop()
        old_ag = origin_class(model, ag_id, 0, 'M', age=1200, home=city_places['home'])
        model.add_new_agent_to_model(old_ag)
        school = city_places['school']
        ckey = np.random.choice(list(school.grouped_studs.keys()))
        school.assign_teacher(old_ag, ckey)
    else:
        ag_id = model.set_available_ids.pop()

        old_ag = origin_class(model, ag_id, 0, 'M', age=100, home=city_places['home'],
                              **{labels[0]: city_places[labels[0]]})
        model.add_new_agent_to_model(old_ag)
    
    city_places['home'].agents_in.add(old_ag)

    if labels[2]:
        grown_ag = old_ag.evolve(new_class, ret_output=True,
                                 **{labels[1]: city_places[labels[1]]})
    else:
        grown_ag = old_ag.evolve(new_class, ret_output=True)

    # check references to old agent are all deleted
    gc.collect()
    assert sys.getrefcount(old_ag) == 2
    if labels[0] and labels[0] != labels[1]:
        assert labels[0] not in grown_ag.loc_info
    if labels[1]:
        assert labels[1] in grown_ag.loc_info


def test_look_for_partner(model):
    for ag in model.schedule.agents:
        if isinstance(ag, Young):
            ag.info['married'] = False
            break
    ag.look_for_partner(avg_years=0.001)
    assert ag.info['married']

            
def test_listen(model):
    pass


@pytest.mark.parametrize("job1, job2, j2_teach", test_data_move_new_home)
def test_move_to_new_home(model, job1, job2, j2_teach):
    """ Test """
    # define home-job pairs for each agent so that each pair is in a different cluster
    h1 = model.geo.clusters_info[0]['homes'][1]
    j1_init = model.geo.clusters_info[0]['jobs'][0] if job1 else None
    h2 = model.geo.clusters_info[1]['homes'][1]
    if not j2_teach:
        j2_init = model.geo.clusters_info[1]['jobs'][0] if job2 else None
    else:
        j2_init = model.geo.clusters_info[1]['schools'][0] if job2 else None

    new_ags_ids = [model.set_available_ids.pop() for _ in range(2)]
    new_ags_age = 1200

    consort1 = Young(model, new_ags_ids[0],
                     1, 'M', age=new_ags_age, home=h1, job=j1_init)
    if j2_teach:
        # TODO: TO BE FIXED
        consort2 = Teacher(model, new_ags_ids[1], 1, 'F', age=new_ags_age, home=h2, job=j2_init)
    else:
        consort2 = Young(model, new_ags_ids[1], 1, 'F', age=new_ags_age, home=h2, job=j2_init)
    
    ags = [consort1, consort2]
    # add agents to model entities
    for ag in ags:
        model.add_new_agent_to_model(ag)
    # update acquaintance status between agents
    consort1.update_acquaintances(consort2, 0)

    # get married
    consort1.get_married(consort2)

    j1_final = consort1.get_current_job()
    j2_final = consort2.get_current_job()
    
    if j1_final == j1_init:
        assert consort2['clust'] == consort1['clust']
        assert consort1 in j1_final.info['employees']
        if j2_init:
            assert consort2 not in j2_init.info['employees']
    if j2_final == j2_init and j2_init:
        assert consort1['clust'] == consort2['clust']
        assert consort1 not in j1_init.info['employees']
        assert consort2 in j2_final.info['employees']


@pytest.mark.parametrize('agent_class', test_data_get_job)
def test_get_job(model, city_places, agent_class):
    ag_id = model.set_available_ids.pop()
    ag = agent_class(model, ag_id, 1, 'M', age=1200, home=city_places['home'])
    model.add_new_agent_to_model(ag)
    # get a job
    ag.get_job()

    if ag.loc_info['job']:
        if type(ag) is Adult:
            assert ag in ag.loc_info['job'].info['employees']
        elif type(ag) is Teacher:
            assert ag in ag.loc_info['job'][0].info['employees']
        elif type(ag) is TeacherUniv:
            assert ag in ag.loc_info['job'][0].info['employees']


@pytest.mark.parametrize("sample_words, mode_type", test_data_update_lang_arrays)
def test_update_lang_arrays(dummy_agent, sample_words, mode_type):

    ag = dummy_agent

    # adapt agent retention levels
    ag.lang_stats['L1']['R'][-100:] = 0.
    # get words whose counting will be updated
    act, act_c = sample_words['L1']
    known_words = np.nonzero(ag.lang_stats['L1']['R'] > 0.1)
    kn_act_bool = np.isin(act, known_words, assume_unique=True)
    act_upd, act_upd_c = act[kn_act_bool], act_c[kn_act_bool]

    wc_init = ag.lang_stats['L1']['wc'][act_upd].copy()

    # check calls through patches
    with patch.object(ag, 'process_unknown_words') as proc_uw:
        with patch.object(ag, 'update_words_memory') as uwm:
            if mode_type == 'speak':
                # speaking treats every word as known by definition: words memory
                # is always updated and unknown-word processing is never invoked,
                # regardless of the agent's current knowledge
                ag.update_lang_arrays(sample_words, mode_type=mode_type)
                assert proc_uw.call_count == 0
                assert uwm.call_count == 1
            elif np.all(kn_act_bool):
                # call method to test
                ag.update_lang_arrays(sample_words, mode_type=mode_type)
                assert proc_uw.call_count == 0
                assert uwm.call_count == 1
            elif np.all(~kn_act_bool):
                proc_uw.return_value = np.array([], dtype=np.int32)
                # call method to test
                ag.update_lang_arrays(sample_words, mode_type=mode_type)
                assert uwm.call_count == 0
            else:
                proc_uw.return_value = np.array([ix for ix, val in enumerate(act)
                                                 if val not in known_words[0]])
                # call method to test
                ag.update_lang_arrays(sample_words, mode_type=mode_type)
                assert proc_uw.call_count == 1
                assert uwm.call_count == 1

    # check updated counts
    if mode_type == 'speak':
        assert np.all(ag.lang_stats['L1']['wc'][act_upd] > wc_init)
    elif mode_type == 'listen':
        assert np.all(ag.lang_stats['L1']['wc'][act_upd] > wc_init)


@pytest.mark.parametrize("sample_words, mode_type", test_data_update_lang_arrays)
def test_process_unknown_words(model, sample_words, mode_type):
    pass


@pytest.mark.parametrize("sample_words, mode_type", test_data_update_lang_arrays)
def test_update_words_memory(model, dummy_agent, sample_words, mode_type):

    agent = dummy_agent

    for lang_label, (act, act_c) in sample_words.items():
        R_init = agent.lang_stats[lang_label]['R'][act].copy()
        S_init = agent.lang_stats[lang_label]['S'][act].copy()
        t_init = agent.lang_stats[lang_label]['t'][act].copy()

        agent.update_words_memory(lang_label, act, act_c)

        assert np.all(agent.lang_stats[lang_label]['R'][act] >= R_init)
        assert np.all(agent.lang_stats[lang_label]['S'][act] > S_init)
        assert np.all(agent.lang_stats[lang_label]['t'][act] <= t_init)


def test_look_for_partner(model):
    pass


# @pytest.mark.parametrize("long", test_data_vocab_choice_model)
# def test_vocab_choice_model(model, long):
#     agent = model.schedule.agents[0]
#     lang = 0 if agent.language == 0 else 1
#     act, act_c = agent.vocab_choice_model(lang, long)
#     assert len(act) > 0
        