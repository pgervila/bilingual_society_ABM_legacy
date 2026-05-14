# Author: Paolo Gervasoni

# IMPORT RELEVANT LIBRARIES
import os
import bisect
import random
import numpy as np
import networkx as nx
# import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pylab as plt
import matplotlib.animation as animation
import pyprind
import deepdish as dd

# IMPORT MESA LIBRARIES ( Model, Grid, Schedule )
from mesa import Model
from mesa.space import MultiGrid
from .schedule import StagedActivationModif

# IMPORT MODEL LIBRARIES
from .agent import Baby, Child, Adolescent, Young, YoungUniv, Teacher, TeacherUniv, Adult, Pensioner
from .geomapping import GeoMapper
from .networks import NetworkBuilder
from .dataprocess import DataProcessor, DataViz

# setting random seed
rand_seed = random.randint(0, 10000)
#rand_seed = 8042
random.seed(rand_seed)
# setting numpy seed
np_seed = np.random.randint(10000)
#np_seed = 5543
np.random.seed(np_seed)

print('rand_seed is {}'.format(rand_seed))
print('np_seed is {}'.format(np_seed))
print('python hash seed is', os.environ.get('PYTHONHASHSEED', 'not set'))


class BiLangModel(Model):

    class _Decorators:
        @classmethod
        def conv_counter(cls, func):
            """
                Decorator that tracks the number of conversations
                each agent has per step
            """
            def wrapper(self, ag_init, others, *args, **kwargs):
                ags = [ag_init]
                ags.extend(others) if (type(others) is list) else ags.append(others)
                for ag in ags:
                    try:
                        ag._conv_counts_per_step += 1
                    except AttributeError:
                        # create attribute if it does not exist yet
                        ag._conv_counts_per_step = 0
                return func(self, ag_init, others, *args, **kwargs)
            return wrapper

    ic_pct_keys = [10, 25, 50, 75, 90]
    family_size = 4
    # TODO : lang policies should be model params, not class attrs
    school_lang_policy = [0, 1]
    jobs_lang_policy = None
    media_lang_policy = None

    steps_per_year = 36
    max_lifetime = 4000
    langs = ('L1', 'L12', 'L21', 'L2')
    similarity_corr = {'L1': 'L2', 'L2': 'L1', 'L12': 'L2', 'L21': 'L1'}
    # avg conversation : 3 min, 20 sec
    # 125 words per minute on average
    # 400 words per avg conversation -> avg conv has 10 tokens if comp_ratio = 40
    num_words_conv = {'VS': 1, 'S': 3, 'M': 10, 'L': 100}

    def __init__(self, num_people, spoken_only=True, width=100, height=100, max_people_factor=5,
                 init_lang_distrib=[0.25, 0.65, 0.1], num_clusters=10, immigration=False, pct_immigration=0.005,
                 lang_ags_sorted_by_dist=True, lang_ags_sorted_in_clust=True, mean_word_distance=0.3,
                 check_setup=False, rand_seed=rand_seed, np_seed=np_seed):
        # TODO: group all attrs in a dict to keep it more tidy
        self.num_people = num_people
        if spoken_only:
            self.vocab_red = 500
        else:
            self.vocab_red = 1000
        self.grid_width = width
        self.grid_height = height
        self.max_people_factor = max_people_factor
        self.init_lang_distrib = init_lang_distrib
        self.num_clusters = num_clusters
        if immigration:
            self.pct_immigration = pct_immigration
        else:
            self.pct_immigration = None
        self.lang_ags_sorted_by_dist = lang_ags_sorted_by_dist
        self.lang_ags_sorted_in_clust = lang_ags_sorted_in_clust
        self.seeds = [rand_seed, np_seed]

        self.conv_length_age_factor = None
        self.death_prob_curve = None

        # define Levenshtein distances between corresponding words of two languages
        self.edit_distances = dict()
        self.edit_distances['original'] = np.random.binomial(10, mean_word_distance, size=self.vocab_red)
        self.edit_distances['mixed'] = np.random.binomial(10, 0.1, size=self.vocab_red)

        # define container for available ids
        self.set_available_ids = set(range(0, max_people_factor * num_people))

        # import lang ICs and lang CDFs data as function of steps. Use directory of executed file
        self.lang_ICs = dd.io.load(os.path.join(os.path.dirname(__file__), 'data', 'init_conds', 'lang_spoken_ics_vs_step.h5'))
        self.cdf_data = dd.io.load(os.path.join(os.path.dirname(__file__), 'data', 'cdfs', 'lang_cdfs_vs_step.h5'))

        # set init mode while building the model
        self.init_mode = True

        # define model grid and schedule
        self.grid = MultiGrid(height, width, False)
        self.schedule = StagedActivationModif(self,
                                              stage_list=["stage_1", "stage_2",
                                                          "stage_3", "stage_4"],
                                              shuffle=True,
                                              shuffle_between_stages=False)

        # instantiate and setup mapping of agents and city objects
        self.geo = GeoMapper(self, num_clusters)
        self.geo.map_model_objects()
        # instantiate and setup networks
        self.nws = NetworkBuilder(self)
        self.nws.build_networks()

        # define datacollector and dataprocessor
        self.data_process = DataProcessor(self)
        # define dataviz
        self.data_viz = DataViz(self)

        # set model curves
        self.set_conv_length_age_factor()
        self.set_death_prob_curve()

        # switch to run mode once model initialization is completed
        self.init_mode = False
        # check model setup if requested
        if check_setup:
            self.check_model_set_up()

    def check_model_set_up(self):
        # check some key configs in model are correct
        for ag in self.schedule.agents:
            if ag.info['age'] > self.steps_per_year:
                if isinstance(ag, Young):
                    if ag.loc_info['job']:
                        if isinstance(ag, Teacher):
                            assert ag['clust'] == ag.loc_info['job'][0].info['clust']
                        else:
                            assert ag['clust'] == ag.loc_info['job'].info['clust']
                else:
                    if ag.loc_info['school']:
                        assert ag['clust'] == ag.loc_info['school'][0].info['clust']
        print('Model is correctly set')

    def set_conv_length_age_factor(self, age_1=14, age_2=65, rate_1=0.0001, rate_3=0.0005,
                                   exp_mult=400):
        """
            Method to compute the correction factor for conversation lengths as a function of age
            The correction factor is a curve with three different sections, defined by the values
            of age_1 and age_2. The middle section defines the plateau value that is equal to one.
            The first section grows from a very small value up to one, according to an S curve whose
            shape is defined by rate_1 and exp_mult. The third section decays from the max value
            according to the exponential rate defined by rate_3.
            Args:
                * age_1: integer. Defines lower key age for slope change in num_words vs age curve
                * age_2: integer. Defines higher key age for slope change in num_words vs age curve
                * rate_1: float. Value to set exponential growth rate for factor in section 1
                * rate_3: float. Value to set exponential decay rate in section 3
                * exp_mult: integer. Value that multiplies exponential function in section 1
            Output:
                * Method sets the factor value as instance attribute. The factor value is a numpy array
                    where values are a function of index (index == age)
        """
        age = np.arange(self.max_lifetime)
        factor = np.zeros(self.max_lifetime)
        # define maximum value for factor
        f = 1.
        # convert pivot ages into steps values
        age_1, age_2 = [self.steps_per_year * age for age in (age_1, age_2)]

        # define sector 1
        decay = -np.log(rate_1 / 100) / age_1
        factor[:age_1] = f + exp_mult * np.exp(- decay * age[:age_1])
        # define sector 2
        factor[age_1:age_2] = f
        # define sector 3
        factor[age_2:] = f - 1 + np.exp(rate_3 * (age[age_2:] - age_2))

        # set factor value
        self.conv_length_age_factor = 1 / factor

    def set_death_prob_curve(self, a=1.23368173e-05, b=2.99120806e-03, c=3.19126705e+01):
        """
            Computes fitted function that provides the death likelihood for a given rounded age
            In order to get the death-probability per step we divide by number of steps per year
            Fitting parameters are from
            https://www.demographic-research.org/volumes/vol27/20/27-20.pdf
            ' Smoothing and projecting age-specific probabilities of death by TOPALS ' by Joop de Beer
            Resulting life expectancy is 77 years and std is ~ 15 years

            Args:
                * a, b, c : float. Fitted model parameters
        """
        self.death_prob_curve = a * (np.exp(b * np.arange(self.max_lifetime)) + c) / self.steps_per_year

    @staticmethod
    def get_newborn_lang(parent1, parent2):
        """
            ALGO to assess which language each parent will speak to newborn child and
            which language category the newborn will have at birth
            Args:
                * parent1: newborn parent agent
                * parent2 : other newborn parent agent
            Output:
                * Method returns 3 integers: newborn lang_type, lang_with_father, lang_with_mother
        """
        # TODO : implement a more elaborated decision process
        # TODO: implement consistence with siblings-parents language
        pcts = np.array(parent1.get_langs_pcts() + parent2.get_langs_pcts())
        langs_with_parents = []
        for pcs, parent in zip([pcts[:2], pcts[2:]], [parent1, parent2]):
            par_lang = parent.info['language']
            if par_lang in [0, 2]:
                lang_with_parent = 0 if par_lang == 0 else 1
            else:
                lang_with_parent = np.random.choice([0, 1], p=pcs / pcs.sum())
            langs_with_parents.append(lang_with_parent)
        lang_with_father = langs_with_parents[0] if parent1.info['sex'] == 'M' else langs_with_parents[1]
        lang_with_mother = langs_with_parents[1] if parent2.info['sex'] == 'M' else langs_with_parents[0]

        langs_with_parents = set(langs_with_parents)
        if langs_with_parents == {0}:
            newborn_lang = 0
        elif langs_with_parents == {0, 1}:
            newborn_lang = 1
        else:
            newborn_lang = 2

        return newborn_lang, lang_with_father, lang_with_mother

    #@_Decorators.conv_counter
    def run_conversation(self, ag_init, others, bystander=None,
                         def_conv_length='M', num_days=10):
        """
            Method that models conversation between ag_init (initiator) and others
            Calls 'get_conv_params' model method to determine conversation parameters
            Then it makes each speaker speak and the rest listen (loop through all involved agents)
            Args:
                * ag_init : agent object instance. Agent that starts conversation
                * others : list of agent class instances. Rest of agents that take part in conversation
                    It can be a single agent object that will be automatically converted into a list
                * bystander: extra agent that may listen to conversation words without actually being involved.
                    Agent vocabulary gets correspondingly updated if bystander agent is specified
                * def_conv_length: string. Default conversation length. Value may be modified
                    depending on agents linguistic knowledge. Values are from keys of 'num_words_conv'
                    model class attribute ('VS', 'S', 'M', 'L')
                * num_days: integer [1, 10]. Number of days in one 10day-step this kind of speech is done
            Output:
                * Method updates lang arrays for all active agents involved. It also updates acquaintances
        """

        # define list of all agents involved in conversation
        ags = [ag_init, others] if (type(others) is not list) else [ag_init, *others]

        # get all parameters of conversation if len(ags) >= 2, otherwise exit method
        try:
            conv_params = self.get_conv_params(ags, def_conv_length=def_conv_length)
        except ZeroDivisionError:
            return
        for ix, (ag, lang) in enumerate(zip(ags, conv_params['lang_group'])):
            if ag.info['language'] != conv_params['mute_type']:
                spoken_words = ag.pick_vocab(lang, conv_length=conv_params['conv_length'],
                                             min_age_interlocs=conv_params['min_group_age'],
                                             num_days=num_days)
                # call listeners' lang arrays updates ( check if there is a bystander)
                listeners = ags[:ix] + ags[ix + 1:] + [bystander] if bystander else ags[:ix] + ags[ix + 1:]
                for listener in listeners:
                    listener.update_lang_arrays(spoken_words, mode_type='listen', delta_s_factor=0.1)
            else:
                # update exclusion counter for excluded agent
                ag.update_lang_exclusion()
        # update acquaintances
        if isinstance(others, list):
            for ix, ag in enumerate(others):
                if ag.info['language'] != conv_params['mute_type']:
                    ag_init.update_acquaintances(ag, conv_params['lang_group'][0])
                    ag.update_acquaintances(ag_init, conv_params['lang_group'][ix + 1])
        else:
            if others.info['language'] != conv_params['mute_type']:
                ag_init.update_acquaintances(others, conv_params['lang_group'][0])
                others.update_acquaintances(ag_init, conv_params['lang_group'][1])

    def get_conv_params(self, ags, def_conv_length='M'): # TODO: add higher thresholds for job conversation
        """
        Method to find out parameters of conversation between 2 or more agents:
            conversation lang or lang spoken by each involved speaker,
            conversation type(mono or bilingual),
            mute agents (agents that only listen),
            conversation length.
        It implements MAXIMIN language rule from Van Parijs
        Args:
            * ags : list of all agent class instances that take part in conversation
            * def_conv_length: string. Default conversation length. Value may be modified
                depending on agents linguistic knowledge. Values are from keys of 'num_words_conv'
                model class attribute ('VS', 'S', 'M', 'L')
        Returns:
            * 'conv_params' dict with following keys and values:
                - lang_group: integer in [0, 1] if unique lang conv or list of integers in [0, 1]
                    if multilingual conversation
                - mute_type: integer. Agent lang type that is unable to speak in conversation
                - multilingual: boolean. True if conv is held in more than one language
                - conv_length: string. Values are from keys of 'num_words_conv'
                    model class attribute ('VS', 'S', 'M', 'L')
                - fav_langs: list of integers in [0, 1].
        """

        # redefine separate agents for readability
        ag_init = ags[0]
        others = ags[1:]

        # set output default parameters
        conv_params = dict(multilingual=False, mute_type=None, conv_length=def_conv_length)

        # get set of language types involved
        ags_lang_types = set([ag.info['language'] for ag in ags])
        # get lists of favorite language per agent
        fav_langs_and_pcts = [ag.get_dominant_lang(ret_pcts=True) for ag in ags]
        # define lists with agent competences and preferences in each language
        fav_lang_per_agent, l_pcts = list(zip(*fav_langs_and_pcts))
        l1_pcts, l2_pcts = list(zip(*l_pcts))

        known_others = [ag for ag in others
                        if ag in self.nws.known_people_network[ag_init]]
        unknown_others = [ag for ag in others
                          if ag not in self.nws.known_people_network[ag_init]]

        def compute_lang_group(default_lang):
            if unknown_others:
                lang_group = default_lang
            else:
                langs_with_known_agents = [self.nws.known_people_network[ag_init][ag]['lang']
                                           for ag in known_others]
                langs_with_known_agents = [lang[0] if isinstance(lang, tuple) else lang
                                           for lang in langs_with_known_agents]
                lang_group = round(sum(langs_with_known_agents) / len(langs_with_known_agents))
            return lang_group

        # define current case
        # TODO: need to save info of how init wanted to talk-> Feedback for AI learning
        if ags_lang_types in [{0}, {0, 1}]:
            conv_params['lang_group'] = compute_lang_group(default_lang=0)
        elif ags_lang_types in [{1, 2}, {2}]:
            conv_params['lang_group'] = compute_lang_group(default_lang=1)
        elif ags_lang_types == {1}:
            # simplified PRELIMINARY NEUTRAL assumption: ag_init will start speaking the language they speak best
            # ( TODO : at this stage no modeling of place bias !!!!)
            # ( TODO: best language has to be compatible with constraints !!!!)
            # who starts conversation matters, but also average lang spoken with already known agents
            lang_init = fav_lang_per_agent[0]
            conv_params['lang_group'] = compute_lang_group(default_lang=lang_init)
        else:
            # monolinguals on both linguistic sides => VERY SHORT CONVERSATION
            # get agents on both lang sides unable to speak in other lang
            idxs_real_monolings_l1 = [idx for idx, pct in enumerate(l2_pcts)
                                      if pct < ags[idx].lang_thresholds['understand']]
            idxs_real_monolings_l2 = [idx for idx, pct in enumerate(l1_pcts)
                                      if pct < ags[idx].lang_thresholds['understand']]

            if not idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # No complete monolinguals on either side
                # All agents partially understand each other langs, but some can't speak l1 and some can't speak l2
                # Conversation is possible when each agent picks their favorite lang
                # TODO: those who can should adapt to init lang
                lang_init = fav_lang_per_agent[0]
                lang_group = tuple([lang_init
                                    if fav_langs_and_pcts[ix][1][lang_init] >= ag.lang_thresholds['speak']
                                    else fav_lang_per_agent[ix]
                                    for ix, ag in enumerate(ags)])
                conv_params.update({'lang_group': lang_group,
                                    'multilingual': True, 'conv_length': 'S'})
            elif idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # There are real L1 monolinguals in the group
                # Everybody partially understands L1, but some agents don't understand L2 at all
                # Some agents only understand and speak L1, while others partially understand but can't speak L1
                # slight bias towards L1 => conversation in L1 (if initiator belongs to this group)
                # but some speakers will stay mute = > short conversation
                mute_type = 2
                if ag_init.info['language'] != mute_type:
                    lang_group = 0
                else:
                    lang_group, mute_type = 1, 0
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'conv_length': 'VS'})

            elif not idxs_real_monolings_l1 and idxs_real_monolings_l2:
                # There are real L2 monolinguals in the group
                # Everybody partially understands L2, but some agents don't understand L1 at all
                # Some agents only understand and speak l2, while others partially understand but can't speak L2
                # slight bias towards l2 => conversation in L2 but some speakers will stay mute = > short conversation
                mute_type = 0
                if ag_init.info['language'] != mute_type:
                    lang_group = 1
                else:
                    lang_group, mute_type = 0, 2
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'conv_length': 'VS'})
            else:
                # There are agents on both lang sides unable to follow other's speech.
                # Initiator agent will speak with whom understands him, others will listen but understand nothing

                if ag_init.info['language'] == 1:
                    # init agent is bilingual
                    # pick majority lang
                    num_l1_speakers = sum([1 if pct >= 0.1 else 0 for pct in l1_pcts])
                    num_l2_speakers = sum([1 if pct >= 0.1 else 0 for pct in l2_pcts])
                    if num_l1_speakers > num_l2_speakers:
                        lang_group, mute_type = 0, 2
                    elif num_l1_speakers < num_l2_speakers:
                        lang_group, mute_type = 2, 0
                    else:
                        lang_group = fav_lang_per_agent[0]
                        mute_type = 2 if lang_group == 0 else 0
                else:
                    # init agent is monolang
                    lang_group = fav_lang_per_agent[0]
                    mute_type = 2 if lang_group == 0 else 0
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'conv_length': 'VS'})

        if not conv_params['multilingual']:
            conv_params['lang_group'] = (conv_params['lang_group'],) * len(ags)

        conv_params['min_group_age'] = min([ag.info['age'] for ag in ags])
        conv_params['fav_langs'] = fav_lang_per_agent

        return conv_params

    def set_lang_ics_in_family(self, family):
        """ Method to set linguistic initial conditions for each member of a family
            Args:
                family: list of 4 agents
        """
        # apply correlation between parents' and children's lang knowledge if parents bilinguals
        # check if at least a parent is bilingual
        if 1 in [m.info['language'] for m in family[:2]]:
            # define list to store bilingual parents' percentage knowledge
            key_parents = []
            for ix_member, member in enumerate(family):
                if ix_member < 2 and member.info['language'] == 1:
                    key = np.random.choice(self.ic_pct_keys)
                    key_parents.append(key)
                    member.set_lang_ics(biling_key=key)
                elif ix_member < 2:
                    lang_mono = member.info['language']
                    member.set_lang_ics()
                elif ix_member >= 2:
                    if len(key_parents) == 1:  # if only one bilingual parent
                        if not lang_mono:  # mono in lang 0
                            key = (key_parents[0] + 100) / 2
                        else:  # mono in lang 1
                            key = key_parents[0] / 2
                    else:  # both parents are bilingual
                        key = sum(key_parents) / len(key_parents)
                    # find index of new key amongst available values
                    idx_key = bisect.bisect_left(self.ic_pct_keys,
                                                 key,
                                                 hi=len(self.ic_pct_keys) - 1)
                    key = self.ic_pct_keys[idx_key]
                    member.set_lang_ics(biling_key=key)
        else:  # monolingual parents
            # check if children are bilingual
            if 1 in [m.info['language'] for m in family[2:]]:
                for ix_member, member in enumerate(family):
                    if ix_member < 2:
                        member.set_lang_ics()
                    else:
                        if member.info['language'] == 1:
                            # logical that child has much better knowledge of parents lang
                            member.set_lang_ics(biling_key=90)
                        else:
                            member.set_lang_ics()
            else:
                for member in family:
                    member.set_lang_ics()

    def get_lang_fam_members(self, family):
        """
            Method to find out lang of interaction btw family members in a 4-members family
            Args:
                * family: list of family agents
            Output:
                * lang_consorts, lang_with_father, lang_with_mother, lang_siblings: tuple of integers
        """
        # language between consorts
        consorts_lang_params = self.get_conv_params([family[0], family[1]])
        lang_consorts = consorts_lang_params['lang_group'][0]
        # language of children with parents
        lang_with_father = consorts_lang_params['fav_langs'][0]
        lang_with_mother = consorts_lang_params['fav_langs'][1]
        # language between siblings
        avg_lang = (lang_with_father + lang_with_mother) / 2
        if avg_lang == 0:
            lang_siblings = 0
        elif avg_lang == 1:
            lang_siblings = 1
        else:
            siblings_lang_params = self.get_conv_params([family[2], family[3]])
            lang_siblings = siblings_lang_params['lang_group'][0]

        return lang_consorts, lang_with_father, lang_with_mother, lang_siblings

    def add_new_agent_to_model(self, agent):
        """
            Method to add a given agent instance to all relevant model entities,
            i.e. grid, schedule, network and clusters info
            Args:
                * agent: agent class instance
            Output:
                * Method adds specified agent to grid, schedule, networks and clusters info
        """

        # Add agent to grid, schedule, network and clusters info
        self.geo.add_agents_to_grid_and_schedule(agent)
        self.nws.add_ags_to_networks(agent)
        self.geo.clusters_info[agent['clust']]['agents'].append(agent)

    def update_friendships(self):
        """ Method to update all agents' friendship bonds """
        for ag in self.schedule.agents:
            try:
                friend_cands = np.random.choice(self.nws.known_people_network[ag],
                                                size=20, replace=False)
            except (ValueError, KeyError):
                friend_cands = []
            for cand in friend_cands:
                try:
                    if ag.check_friend_conds(cand, min_times=10):
                        ag.make_friend(cand)
                        break
                except AttributeError:
                    break

    def add_immigration_family(self, lang, clust_ix):
        """ Method to add an immigration family
            Args:
                * lang: integer in [0, 1, 2]
                * clust_ix: integer. Index of cluster where family will live
        """
        family_langs = [lang] * self.family_size
        family = self.geo.create_new_family(family_langs)
        parent = family[0]
        # find random job
        job = random.choice(self.geo.clusters_info[clust_ix]['jobs'])
        job.hire_employee(parent, move_home=False, ignore_lang_constraint=True)
        # assign same home to all family members
        fam_home = parent.find_home(criteria="close_to_job")
        fam_home.assign_to_agent(family)
        for agent in family:
            self.add_new_agent_to_model(agent)
        self.nws.define_family_links(family)
        # find closest school to home
        for child in family[2:]:
            child.register_to_school()

    def add_immigration(self):
        """ Method to add a random number of immigration families per year"""
        expected_num_fam = int(self.pct_immigration * self.schedule.get_agent_count() /
                               self.family_size)
        num_fam = sum([random.random() < (expected_num_fam / self.steps_per_year)
                       for _ in range(self.steps_per_year)])
        for fam in range(num_fam):
            # the larger the cluster the more likely that it attracts immigration
            prob = self.geo.cluster_sizes / self.geo.cluster_sizes.sum()
            clust_ix = np.random.choice(self.num_clusters, p=prob)
            self.add_immigration_family(0, clust_ix)

    def remove_from_locations(self, agent, replace=False, grown_agent=None, upd_course=False):
        """
            Method to remove agent instance from locations in agent 'loc_info' dict attribute
            Replacement by grown_agent will depend on self type
            Args:
                * agent: agent instance to be removed
                * replace: boolean. True if agent has to be replaced. Default False
                * grown_agent: agent instance. It must be specified in case 'replace'
                    is set to True
                * upd_course: boolean. True if removal action is because of periodic updating
                every academic year. Only applies to removal from school or university. Default False.
            Output:
                * agent removed from all instances it belonged to. If replace is set to True,
                    agent is replaced in all locations by specified 'grown_agent'
        """
        # TODO : should it be a geomapping method ?

        # remove old instance from cluster (and replace with new one if requested)
        if replace:
            self.geo.update_agent_clust_info(agent, agent.loc_info['home'].info['clust'],
                                             update_type='replace', grown_agent=grown_agent)
        else:
            self.geo.update_agent_clust_info(agent, agent.loc_info['home'].info['clust'])

        # remove agent from all locations where it belongs to
        for key in agent.loc_info:
            if key == 'school':
                school = agent.loc_info['school'][0]
                if isinstance(agent, (Baby, Child)):
                    try:
                        school.remove_student(agent, replace=replace, grown_agent=grown_agent)
                    except AttributeError:
                        continue
                elif isinstance(agent, Adolescent):
                    # if Adolescent, agent will never be replaced at school by grown agent
                    school.remove_student(agent, upd_course=upd_course)
            elif key == 'home':
                home = agent.loc_info['home']
                home.remove_agent(agent, replace=replace, grown_agent=grown_agent)
            elif key == 'job':
                if isinstance(agent, TeacherUniv):
                    try:
                        univ, course_key, fac_key = agent.loc_info['job']
                        if isinstance(grown_agent, Pensioner):
                            univ.faculties[fac_key].remove_employee(agent)
                        else:
                            univ.faculties[fac_key].remove_employee(agent, replace=replace,
                                                                    new_teacher=grown_agent)
                    except TypeError:
                        pass
                elif isinstance(agent, Teacher):
                    try:
                        school, course_key = agent.loc_info['job']
                        if isinstance(grown_agent, Pensioner):
                            school.remove_employee(agent)
                        else:
                            school.remove_employee(agent, replace=replace, new_teacher=grown_agent)
                    except TypeError:
                        pass
                elif isinstance(agent, Adult):
                    # if Adult, remove from job without replacement
                    job = agent.loc_info['job']
                    if job: job.remove_employee(agent)
                elif isinstance(agent, Young):
                    # if Young, remove from job with replacement
                    job = agent.loc_info['job']
                    if job: job.remove_employee(agent, replace=replace, new_agent=grown_agent)
            elif key == 'university':
                univ, course_key, fac_key = agent.loc_info['university']
                univ.faculties[fac_key].remove_student(agent, upd_course=upd_course)

    def remove_after_death(self, agent):
        """
            Removes agent object from all places where it belongs.
            It makes sure no references to agent object are left after removal,
            so that garbage collector can free memory
            Call this function if death conditions for agent are verified
        """
        # if dead agent was married, update marriage attribute from consort
        if isinstance(agent, Young) and agent.info['married']:
            consort = agent.get_family_relative('consort')
            consort.info['married'] = False

        # Remove agent from all networks
        for network in [self.nws.family_network,
                        self.nws.known_people_network,
                        self.nws.friendship_network]:
            try:
                network.remove_node(agent)
            except nx.NetworkXError:
                continue
        # remove agent from grid and schedule
        self.grid._remove_agent(agent.pos, agent)
        self.schedule.remove(agent)
        # remove agent from all locations where it belonged to
        self.remove_from_locations(agent)
        # make id from deceased agent available
        # self.set_available_ids.add(agent.unique_id)

    def step(self):
        self.schedule.step()
        self.data_process.collect()
        #print('Completed step number {}'.format(self.schedule.steps))

    def update_centers(self):
        """ Method to update students, teachers and courses
            at the end of each year, as well as jobs' language policy
        """
        for clust_idx, clust_info in self.geo.clusters_info.items():
            if 'university' in clust_info:
                for fac in clust_info['university'].faculties.values():
                    if fac.info['students']:
                        fac.update_courses_phase_1()
            for school in clust_info['schools']:
                school.update_courses_phase_1()
            # set lang policy in job centers
            for job in clust_info['jobs']:
                job.set_lang_policy()
        for clust_idx, clust_info in self.geo.clusters_info.items():
            if 'university' in clust_info:
                for fac in clust_info['university'].faculties.values():
                    if fac.info['students']:
                        fac.update_courses_phase_2()
            for school in clust_info['schools']:
                school.update_courses_phase_2()
                # every 4 years only, make teachers swap
                if not self.schedule.steps % (4 * self.steps_per_year):
                    school.swap_teachers_courses()
                # TODO: check if courses have more than 25 students enrolled => Create new school

    def run_model(self, steps, save_data_freq=50, pickle_model_freq=5000,
                  viz_steps_period=None, save_dir=''):
        """ Run model and save frames if required
            Args
                * steps: integer. Total steps to run
                * save_data_freq: int. Frequency of model data saving as measured in steps
                * pickle_model_freq: int. Frequency of model pickling as measured in steps
                * viz_steps_period : integer. Save frames every specified number of steps
                * save_dir : string. It specifies directory where frames will be saved
        """
        pbar = pyprind.ProgBar(steps)
        self.save_dir = save_dir
        if viz_steps_period:
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, save_dir)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
        for _ in range(steps):
            self.step()
            if not self.schedule.steps % save_data_freq:
                self.data_process.save_model_data(save_data_freq)
            if not self.schedule.steps % pickle_model_freq:
                self.data_process.pickle_model()
            if viz_steps_period:
                if not self.schedule.steps % viz_steps_period:
                    self.data_viz.show_results(step=self.schedule.steps,
                                               plot_results=False, save_fig=True)
            pbar.update()

    def run_and_animate(self, steps, plot_type='imshow'):
        fig = plt.figure()
        grid_size = (3, 5)
        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        ax1.set_xlim(0, steps)
        ax1.set_ylim(0, 1)
        ax1.xaxis.tick_bottom()
        line10, = ax1.plot([], [], lw=2, label='count_spa', color='darkblue')
        line11, = ax1.plot([], [], lw=2, label='count_bil', color='g')
        line12, = ax1.plot([], [], lw=2, label='count_cat', color='y')
        ax1.tick_params('x', labelsize='small')
        ax1.tick_params('y', labelsize='small')
        ax1.legend(loc='best', prop={'size': 8})
        ax1.set_title("lang_groups")
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        ax2.set_xlim(0, steps)
        ax2.set_ylim(0, self.max_people_factor * self.num_people)
        line2, = ax2.plot([], [], lw=2, label = "total_num_agents", color='k')
        ax2.tick_params('x', labelsize='small')
        ax2.tick_params('y', labelsize='small')
        ax2.legend(loc='best', prop={'size': 8})
        ax2.set_title("num_agents")
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        ax3.set_xlim(0, steps)
        ax3.set_ylim(0, 1)
        ax3.tick_params('x', labelsize='small')
        ax3.tick_params('y', labelsize='small')
        line3, = ax3.plot([], [], lw=2, label='biling_evol')
        ax3.legend(loc='best', prop={'size': 8})
        ax3.set_title("biling_quality")
        ax4 = plt.subplot2grid(grid_size, (0, 0), rowspan=3, colspan=3)
        ax4.set_xlim(0, self.grid_width-1)
        ax4.set_ylim(0, self.grid_height-1)
        if plot_type == 'imshow':
            im_2D = ax4.imshow(np.zeros((self.grid_width, self.grid_height)),
                               vmin=0, vmax=2, cmap='viridis',
                               interpolation='nearest', origin='lower')
            fig.colorbar(im_2D)
        elif plot_type == 'scatter':
            dots = ax4.scatter([], [], c=[], vmin=0, vmax=2, cmap='viridis')
            fig.colorbar(dots)
        time_text = ax4.text(0.02, 0.95, '', transform=ax4.transAxes)

        def init_show():
            if plot_type == 'imshow':
                im_2D.set_array(np.random.choice([np.nan, 0], p=[1, 0], size=(self.grid_width, self.grid_height)))
                return im_2D,
            elif plot_type == 'scatter':
                dots.set_offsets([0,0])
                return dots,

        def run_and_update(i):
            #run model step
            self.step()

            #create plots and data for 1D plots
            data = self.data_process.get_model_vars_dataframe()
            line10.set_data(data.index, data['count_spa'])
            line11.set_data(data.index, data['count_bil'])
            line12.set_data(data.index, data['count_cat'])

            line2.set_data(data.index, data['total_num_agents'])

            line3.set_data(data.index, data['biling_evol'])
            # generate data for 2D representation
            self.create_agents_attrs_data('language')
            # create 2D plot
            time_text.set_text('time = %.1f' % i)
            if plot_type == 'imshow':
                im_2D.set_array(self.df_attrs_avg.unstack('x'))
                return line10, line11, line12, line2, line3, im_2D, time_text
            else:
                data = np.hstack((self.df_attrs_avg.reset_index()['x'][:, np.newaxis],
                                  self.df_attrs_avg.reset_index()['y'][:, np.newaxis]))
                dots.set_offsets(data)
                dots.set_array(self.df_attrs_avg.reset_index()['values'])
                return line10, line11, line12, line2, line3, dots, time_text

        # generate persistent animation object
        ani = animation.FuncAnimation(fig, run_and_update, init_func=init_show,
                                      frames=steps, interval=100, blit=True, repeat=False)
        #plt.tight_layout()
        plt.show()




