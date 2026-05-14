# IMPORT LIBS
import random
import string
from math import ceil
from collections import defaultdict
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from numba import njit, prange

# Import private library to model lang zipf CDF
from .zipf_generator import randZipf


@njit
def numba_comp_delta_S(a, b, c, d, e, f, g):
    return a * (b * (e * c * np.exp(f * 100 * d) + g))


@njit
def numba_comp_R(a, b, c):
    return np.exp(a * b / c)


@njit
def numba_comp_max(a, b):
    return np.maximum(a, b)


@njit
def numba_comp_exp(a, b):
    return np.power(a, b)


@njit
def numba_counter(a, b):
    c = 0
    for i in prange(len(a)):
        if a[i] > b:
            c += 1
    return c


@njit
def numba_in1d(ar1, ar2, ret, bool_ar):
    ar = np.concatenate((ar1, ar2))
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    bool_ar[:-1] = (sar[1:] == sar[:-1])
    for i, e in enumerate(order):
        ret[e] = bool_ar[i]
    return ret[:len(ar1)]


class BaseAgent:
    """ Basic agent class that contains attributes and
        methods common to all lang agents subclasses
        Args:
            * model: model class instance
            * unique_id: integer. Value must be unique because it is used to construct
                the instance hash value. It should never be modified. Value should be drawn
                from available values in model.set_available_ids
            * language: integer in [0, 1, 2]. 0 -> mono L1, 1 -> bilingual, 2 -> mono L2
            * sex: string. Either 'M' or 'F'
            * age: integer
            * home: home class instance
            * lang_act_thresh: float. Minimum percentage of known words to be able
                to ACTIVELY communicate in a given language at a basic level.
                This percentage is not sufficient to be considered bilingual
                if more than one language is spoken (see update_lang_switch method).
            * lang_passive_thresh: float. Minimum percentage of known words to be able
                to PASSIVELY understand a given language
            * import_ic: boolean. True if linguistic initial conditions must be imported
    """

    # define memory retrievability constant
    k = np.log(10 / 9)
    max_life_steps = 3600

    def __init__(self, model, unique_id, language, sex, age=0, home=None, lang_act_thresh=0.1,
                 lang_passive_thresh=0.025, import_ic=False):
        """ language => 0, 1, 2 => spa, bil, cat  """
        self.model = model
        self.unique_id = unique_id
        # generate random number of maxim friends with fat-tail distribution
        max_num_friends = np.clip(np.random.lognormal(1, 1), a_min=3, a_max=None).astype(int)
        self.info = {'age': age, 'language': language, 'sex': sex, 'max_num_friends': max_num_friends }
        self.loc_info = {'home': None}
        if home:
            home.assign_to_agent(self)
        self.lang_thresholds = {'speak': lang_act_thresh, 'understand': lang_passive_thresh}

        # define container for languages' tracking and statistics
        self.lang_stats = defaultdict(dict)

        # define mask for each step
        self.step_mask = {lang: np.zeros(self.model.vocab_red, dtype=bool)
                          for lang in self.model.langs}
        if import_ic:
            self.set_lang_ics()
        else:
            # set null knowledge in all possible langs
            for lang in self.model.langs:
                self._set_null_lang_attrs(lang)
        # initialize word counter
        self.wc_init = dict()
        self.wc_final = dict()
        for lang in self.model.langs:
            self.wc_init[lang] = np.zeros(self.model.vocab_red, dtype=np.int64)
            self.wc_final[lang] = np.zeros(self.model.vocab_red, dtype=np.int64)
        # initialize conversation counter for data collection
        self._conv_counts_per_step = 0

    def _set_null_lang_attrs(self, lang, S_0=0.01, t_0=1000):
        """Private method that sets null linguistic knowledge in specified language, i.e. no knowledge
           at all of it
           Args:
               * lang: string. It can take two different values: 'L1' or 'L2'
               * S_0: float. Initial value of memory stability
               * t_0: integer. Initial value of time-elapsed (in days) from last time words were encountered
        """
        self.lang_stats[lang]['S'] = np.full(self.model.vocab_red, S_0, dtype=float)
        self.lang_stats[lang]['t'] = np.full(self.model.vocab_red, t_0, dtype=float)
        self.lang_stats[lang]['R'] = np.exp(- self.k *
                                            self.lang_stats[lang]['t'] /
                                            self.lang_stats[lang]['S']
                                            ).astype(np.float32)
        self.lang_stats[lang]['wc'] = np.zeros(self.model.vocab_red, dtype=np.int64)
        self.lang_stats[lang]['pct'] = np.zeros(self.max_life_steps, dtype=np.float64)
        self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                          len(self.model.cdf_data['s'][self.info['age']]))
        # memory effort per compressed word
        self.set_memory_effort_per_word(lang)
        # conversation failure counter
        self.lang_stats[lang]['excl_c'] = np.zeros(self.max_life_steps, dtype=np.float64)
        # set memory weights to evaluate lang exclusion
        self.set_excl_weights()

    def set_lang_ics(self, s_0=0.01, t_0=1000, biling_key=None):
        """ Set agent's linguistic Initial Conditions.

        The legacy per-age IC file (lang_spoken_ics_vs_step.h5) has been dropped.
        All agents start from null linguistic knowledge; a realistic demography is
        produced by the model's warmup phase instead (see BiLangModel warmup_steps).
        `biling_key` is accepted for call-site compatibility but is currently a
        no-op -- reintroducing graded initial bilingualism is deferred to the
        model-improvement plan.
        """
        for lang in ('L1', 'L2', 'L12', 'L21'):
            self._set_null_lang_attrs(lang, s_0, t_0)
        # set weights to model reaction to linguistic exclusion
        self.set_excl_weights()

    def set_memory_effort_per_word(self, lang, min_num_t=3, max_num_t=7):
        """
            Method that assigns a minimum number of times to each compressed token
            before it can be encoded in memory. Since one compressed token corresponds to 40 real words,
            it takes repetition to start memorizing this information. Since the
            word flow is also compressed, the modeled minimum number of times to memorize a token
            is also compressed as compared to the real one ( 3 repetitions in compresed vocabulary
            corresponds to comp_f * 3 (40 * 3 = 120) repetitions in real vocabulary )
            Args:
                * lang: string. Language
                * min_num_t: integer. Minimum number of interactions with a compressed token
                    to start remembering it
                * max_num_t: integer. Maximum number of interactions with a compressed token
                    to start remembering it
            Output:
                * sets value of self.lang_stats[lang]['mem_eff'] ( numpy array of integers )
        """
        # m = 1 / self.model.vocab_red
        # self.lang_stats[lang]['mem_eff'] = (m * np.arange(self.model.vocab_red)**1 + min_num_t).astype(int)

        self.lang_stats[lang]['mem_eff'] = np.random.randint(min_num_t, max_num_t,
                                                             size=self.model.vocab_red)

    def set_excl_weights(self, mem_window_length=5):
        """ Method to set values for weights to assess degree of exclusion
            in recent steps. Weights are used o compute an EMA together with
            the sequence of the """
        x = np.linspace(-1, 0, mem_window_length)
        weights = np.exp(x)
        weights /= weights.sum()
        self.info['excl_weights'] = weights

    def get_langs_pcts(self, lang=None):
        """
            Method that returns pct knowledge in L1 and L2
            Args:
                * lang: integer. Value from [0, 1]
            Output:
                * Percentage knowledge in specified lang or
                  percentage knowledges in both langs if lang arg is
                  not specified
        """
        if lang is None:
            pct_lang1 = self.lang_stats['L1']['pct'][self.info['age']]
            pct_lang2 = self.lang_stats['L2']['pct'][self.info['age']]
            return pct_lang1, pct_lang2
        else:
            lang = 'L1' if lang == 0 else 'L2'
            return self.lang_stats[lang]['pct'][self.info['age']]

    def get_dominant_lang(self, ret_pcts=False):
        """
            Method that returns dominant language after
            computing pct knowledges in each language
        """
        pcts = self.get_langs_pcts()
        if pcts[0] != pcts[1]:
            dominant_lang = np.argmax(pcts)
        else:
            dominant_lang = 1 if random.random() >= 0.5 else 0
        if ret_pcts:
            return dominant_lang, pcts
        else:
            return dominant_lang

    def update_lang_switch(self, switch_threshold=0.2):
        """
           Switch to a new linguistic regime when threshold is reached
           If language knowledge falls below switch_threshold value, agent
           becomes monolingual
           This method is called from schedule at each step
           Args:
                * switch_threshold: float. Threshold value below which agent
                is no longer considered fluent in that language. It defaults to 0.2
        """

        # TODO : adapt all languages spoken to known people when threshold reaches 0.1

        if self.info['language'] == 0:
            if self.get_langs_pcts(1) >= switch_threshold:
                self.info['language'] = 1
        elif self.info['language'] == 2:
            if self.get_langs_pcts(0) >= switch_threshold:
                self.info['language'] = 1
        elif self.info['language'] == 1:
            if self.get_langs_pcts(0) < switch_threshold:
                self.info['language'] = 2
            elif self.get_langs_pcts(1) < switch_threshold:
                self.info['language'] = 0

    def grow(self, growth_inc=1):
        """ Convenience method to update agent age at each step.

        Age is clamped to max_life_steps - 1 so it can never index past the
        per-age arrays (pct, excl_c, cdf rows), all of which have length
        max_life_steps. At this age the death-probability curve makes survival
        astronomically unlikely; the clamp is a guard against an off-by-one.
        """
        self.info['age'] = min(self.info['age'] + growth_inc, self.max_life_steps - 1)

    def evolve(self, new_class, ret_output=False, upd_course=False):
        """ It replaces current agent with a new agent subclass instance.
            It removes current instance from all networks, lists, sets in model and
            replaces it with the new_class instance.
            Args:
                * new_class: class. Agent subclass that will replace the current one
                * ret_output: boolean. True if grown_agent needs to be returned as output
                * upd_course: boolean. True if evolution involves agent quitting school or univ.
                    Default False
        """
        # create new_agent instance
        grown_agent = new_class(self.model, self.unique_id, self.info['language'], self.info['sex'])

        # copy all current instance attributes to the new agent instance
        for key, val in self.__dict__.items():
            setattr(grown_agent, key, val)
        # TODO: explore __slots__ approach for agent class ( getattr(self, key) in self.__slots__ )

        # replace agent node in networks
        relabel_key = {self: grown_agent}
        for network in [self.model.nws.family_network,
                        self.model.nws.known_people_network,
                        self.model.nws.friendship_network]:
            try:
                nx.relabel_nodes(network, relabel_key, copy=False)
            except (KeyError, nx.NetworkXError):
                continue

        # replace agent in grid
        self.model.grid._remove_agent(self.pos, self)
        self.model.grid.place_agent(grown_agent, self.pos)
        # replace agent in schedule
        self.model.schedule.replace_agent(self, grown_agent)
        # remove and replace agent from ALL locations where it belongs to
        self.model.remove_from_locations(self, replace=True,
                                         grown_agent=grown_agent, upd_course=upd_course)

        if ret_output:
            return grown_agent

    def random_death(self, ret_out=False):
        """
            Method to randomly determine agent death or survival at each step
            Args:
                * ret_out: boolean. True if random death boolean result needs to be returned in case of death
            Output:
                * If random event results into agent death, method calls 'remove_after_death' method.
                    Otherwise no action is performed
        """
        if random.random() < self.model.death_prob_curve[self.info['age']]:
            self.model.remove_after_death(self)
            if ret_out:
                return True

    def get_family_relative(self, fam_link):
        """
            Retrieve agent instance from family network
            that satisfies family link
            Input:
                * fam_link: string. String must be an edge label
                as defined in family network
        """
        if fam_link not in ['mother', 'father', 'consort']:
            return [ag for ag, labels in self.model.nws.family_network[self].items()
                    if labels['fam_link'] == fam_link]
        else:
            for ag, labels in self.model.nws.family_network[self].items():
                if labels['fam_link'] == fam_link:
                    return ag

    def go_back_home(self):
        self.model.grid.move_agent(self, self.loc_info['home'].pos)

    @classmethod
    def method_calls_counter(cls, func):
        """ Decorator that counts the number of times
            an agent method is called per step """
        def wrapper(self, *args, **kwargs):
            step = self.model.schedule.steps
            f_name = func.__name__
            if not hasattr(self, '_method_call_counts'):
                self._method_call_counts = defaultdict(list)
            if not self._method_call_counts[f_name]:
                self._method_call_counts[f_name].append(0)
            elif len(self._method_call_counts[f_name]) <= step:
                self._method_call_counts[f_name].append(0)
            self._method_call_counts[f_name][step] += 1
            return func(self, *args, **kwargs)
        return wrapper

    @classmethod
    def words_per_conv_counter(cls, func):
        def wrapper(self, *args, **kwargs):
            step = self.model.schedule.steps
            if not hasattr(self, '_words_per_conv_counts'):
                self._words_per_conv_counts = defaultdict(lambda: defaultdict(list))
            for time, stage in zip(['.0', '.25', '.5', '.75'],
                                   ['stage_1', 'stage_2', 'stage_3', 'stage_4']):
                if time in str(float(self.model.schedule.time)):
                    break
            self._words_per_conv_counts[step][stage].append((kwargs['conv_length'],
                                                             kwargs['num_days']))
            return func(self, *args, **kwargs)
        return wrapper

    @classmethod
    def conv_counter(cls, func):
        """ Decorator that tracks the number of conversations
            each agent has per step """
        def wrapper(self, group, *args, **kwargs):
            ags = [self] + group
            for ag in ags:
                try:
                    ag._conv_counts_per_step += 1
                except AttributeError:
                    ag._conv_counts_per_step = 0
            return func(self, group, *args, **kwargs)
        return wrapper

    def __getitem__(self, clust):
        return self.loc_info['home'].info[clust]

    def __hash__(self):
        return hash(self.unique_id)

    def __eq__(self, other):
        if not isinstance(other, BaseAgent):
            return NotImplemented
        return self.unique_id == other.unique_id and self.__class__ == other.__class__

    def __repr__(self):
        home = self.loc_info['home']
        if home:
            return "".join([type(self).__name__,
                            "_{0.unique_id!r}_clust{1!r}"]).format(self,
                                                                   home.info['clust'])
        else:
            return "".join([type(self).__name__, "_{0.unique_id!r}"]).format(self)


class ListenerAgent(BaseAgent):
    """ BaseAgent class augmented with listening-related methods """

    #@BaseAgent.method_calls_counter
    def listen(self, to_agent=None, min_age_interlocs=None, num_days=10):
        """
            Method to listen to conversations, media, etc... and update corresponding vocabulary
            Args:
                * to_agent: class instance (optional). It can be either a language agent or a media agent.
                    If not specified, self agent will listen to a random conversation taking place on his cell
                * min_age_interlocs: integer. Allows to adapt vocabulary choice to the youngest
                    participant in the conversation
                * num_days: integer. Number of days per step in which listening action takes place on average.
                    A step equals 10 days.
        """
        if not to_agent:
            # TODO: agents should know each other
            # get all non-Baby agents currently placed on chosen cell
            x, y = self.pos
            others = {ag for ag in self.model.grid[x][y] if type(ag) != Baby}
            others.remove(self)
            # if two or more agents in cell, conversation is possible
            if len(others) >= 2:
                conv_ags = np.random.choice(list(others), size=2, replace=False)
                # call run conversation with bystander
                self.model.run_conversation(conv_ags[0], conv_ags[1],
                                            bystander=self, def_conv_length='VS')
        else:
            # make other agent speak and 'self' agent get the listened vocab
            if self in self.model.nws.known_people_network[to_agent]:
                lang = self.model.nws.known_people_network[to_agent][self]['lang']
            else:
                conv_params = self.model.get_conv_params([to_agent, self])
                if conv_params['multilingual']:
                    lang = conv_params['lang_group'][0]
                else:
                    lang = conv_params['lang_group']
            spoken_words = to_agent.pick_vocab(lang, conv_length='S', min_age_interlocs=min_age_interlocs,
                                               num_days=num_days)
            # update listener's lang arrays
            self.update_lang_arrays(spoken_words, mode_type='listen', num_days=num_days)

    def listen_to_media(self):
        """ Method to listen languages in mass media: internet, radio, TV, movies """
        # TODO : implement 'listen to media' option
        pass

    def update_lang_arrays(self, sample_words, mode_type='speak', delta_s_factor=0.1, pct_threshold=0.9,
                           pct_threshold_und=0.1, max_edit_dist=2, min_prob_und=0.1, num_days=1, learning=False):
        """
            Method to compute and update main arrays that define agent linguistic knowledge
            Args:
                * sample_words: dict where keys are lang labels and values are tuples of
                    2 NumPy integer arrays. First array is of conversation-active unique word indices,
                    second is of corresponding counts of those words
                * mode_type: string. Defines type of agent action with words: 'speak', 'listen', 'read', 'media'
                * delta_s_factor: positive float < 1.
                    Defines increase of mem stability due to passive rehearsal
                    as a fraction of that due to active rehearsal
                * min_mem_times: positive integer. Minimum number of times to start remembering a word.
                    Since one compressed token correspond to 40 real words, it takes repetition
                    to start memorizing this information. It should be understood as average time it takes
                    for a word meaning to be incorporated in memory
                * pct_threshold: positive float < 1. Value to define lang knowledge in percentage.
                    If retrievability R for a given word is higher than pct_threshold,
                    the word is considered as well known. Otherwise, it is not
                * pct_threshold_und : positive float < 1. If retrievability R for a given word
                    is higher than pct_threshold, the word can be correctly understood.
                    Otherwise, it cannot
                * max_edit_dist : integer. Maximum value of edit distance between heard/read
                    and known languages in order to recognise an unknown word. For values equal or higher than
                    max_edit_dist, unknown words cannot be recognised.
                * min_prob_und: 0 < float < 1 . Minimum probability to understand a fraction of conversation
                * num_days: integer. Number of times per step conversation takes place
                * learning: boolean. True if agent is studying / learning words. Default False.
        """

        for lang, (act, act_c) in sample_words.items():

            # UPDATE WORD COUNTING + pre-processing for S, t, R UPDATE

            # If words are from listening/reading/media, they might be new to agent
            # ds_factor value will depend on action type

            if mode_type in ['listen', 'media', 'read']:
                # get all already-known words by agent in global reference system
                known_words = np.nonzero(self.lang_stats[lang]['R'] > pct_threshold_und)[0]
                # Of all active words, how many are already known by agent ?? =>
                # boolean mask of known-active words projected on active words reference system
                ret = np.zeros(len(act) + len(known_words), dtype=bool)
                bool_ar = np.zeros(len(act) + len(known_words), dtype=bool)
                kn_act_bool = numba_in1d(act, known_words, ret, bool_ar)
                # get indices of known words on active words reference system
                kn_words_idxs = np.where(kn_act_bool)[0]
                # update counting of known words
                self.lang_stats[lang]['wc'][act[kn_words_idxs]] += act_c[kn_words_idxs]
                # check if all active words in conversation are known
                if not np.all(kn_act_bool):
                    # find indices, if any, of new words whose memory will be updated
                    if not learning:
                        new_words_idxs = self.process_unknown_words(lang, act, act_c, kn_act_bool,
                                                                    max_edit_dist=max_edit_dist,
                                                                    min_prob_und=min_prob_und,
                                                                    num_days=num_days)
                    else:
                        new_words_idxs = self.learn_unknown_words(lang, act, act_c, kn_act_bool)
                    if new_words_idxs is not None:
                        upd_idxs = np.concatenate((kn_words_idxs, new_words_idxs))
                        # words whose memory will be updated and corresponding counts
                        act, act_c = act[upd_idxs], act_c[upd_idxs]
                    else:
                        act, act_c = act[kn_words_idxs], act_c[kn_words_idxs]
                ds_factor = delta_s_factor
            else:
                # when speaking, all words are known by definition
                # update word counts
                self.lang_stats[lang]['wc'][act] += act_c
                ds_factor = 1

            # check if there are any words for S, t, R update
            if act.size:
                self.update_words_memory(lang, act, act_c, ds_factor, pct_threshold)

    def process_unknown_words(self, lang, act, act_c, kn_act_bool, max_edit_dist=2,
                              pct_threshold_und=0.1, min_prob_und=0.1, num_days=1):
        """
            Method to track recognised words among unknown ones,
            or to randomly single out a new word out of unknown ones.
            Ability to recognise new words depends on language similarity array.
            Probability to single out a new word depends on degree of language knowledge.
            Method updates counting of recognised or identified new words, and
            returns indices of new words whose memory will be updated
            Args:
                * lang: string. Label that identifies language ('L1', 'L2', 'L12', 'L21')
                * act, act_c: numpy arrays. Indices and number of occurrences of active words in conversation
                * kn_act_bool: numpy array of booleans. Values are True if word is known, False otherwise.
                    Expressed on active words base
                * max_edit_dist: int. For values of edit distance higher or equal to this value,
                    words cannot be recognised without previous knowledge
                * pct_threshold_und: float. Minimum retention value to consider a word as already known
                * min_prob_und: 0 < float < 1 . Minimum probability to understand a fraction of conversation
                    regardless of language knowledge. Must be different than zero to allow learners to catch new words
                * num_days: integer. Number of times per step conversation takes place
            Output:
                * new_words_idxs: numpy array. Indices of unknown words whose memory can be updated,
                    EXPRESSED on active words coordinates (to be processed by 'update_lang_arrays_method')
        """

        # set default value for indices of new words.
        new_words_idxs = None

        idx_ukn_act = np.where(~kn_act_bool)[0] # indices of unknown words on active coordinates
        val_ukn_act = act[idx_ukn_act] # unknown words values ( indices on global coordinates == the words themselves)

        # TODO : word recognition and integration of new words should not be excluding actions
        # find recognised words: words SIMILAR ENOUGH (cond1) to those ALREADY KNOWN (cond2) in other language
        other_lang = self.model.similarity_corr[lang]
        if lang in ['L1', 'L2']:
            rec_cond1 = self.model.edit_distances['original'][val_ukn_act] < max_edit_dist
        else:
            rec_cond1 = self.model.edit_distances['mixed'][val_ukn_act] < max_edit_dist
        rec_cond2 = self.lang_stats[other_lang]['R'][val_ukn_act] > pct_threshold_und
        # Since both conditions are expressed on act words ref system
        rec_cond = rec_cond1 * rec_cond2
        val_ukn_act_rec, idx_ukn_act_rec = val_ukn_act[rec_cond], idx_ukn_act[rec_cond]

        # check if there are recognised words
        if val_ukn_act_rec.size:
            # update counting of recognised words
            self.lang_stats[lang]['wc'][val_ukn_act_rec] += np.ones(len(idx_ukn_act_rec), dtype=np.uint8)
            # find words whose memory can be updated among recognised ones
            upd_cond = self.lang_stats[lang]['wc'][val_ukn_act_rec] >= self.lang_stats[lang]['mem_eff'][val_ukn_act_rec]
            idx_ukn_act_upd = idx_ukn_act_rec[upd_cond]
            if idx_ukn_act_upd.size:
                new_words_idxs = idx_ukn_act_upd
        else:
            # Probability that a new unknown word is identified increases with lang knowledge
            # Define proportion of tokens understood in conversations.
            # The higher this proportion is, the more likely that a new word is identified
            pct_understood = (kn_act_bool.sum() / kn_act_bool.size)
            # set minimum probability different than zero to allow beginners to learn
            pct_understood = min_prob_und if pct_understood <= min_prob_und else pct_understood
            # get max number new counts that can be POTENTIALLY identified
            max_num_pot_new_words = np.random.binomial(num_days, pct_understood)
            # compute counts of unknown words in conversation
            wc_ukn_words = act_c[idx_ukn_act]
            # get max number new counts that can be EFFECTIVELY identified
            max_num_new_words = min(max_num_pot_new_words, wc_ukn_words.sum())

            # compute probabs of each new token to be incorporated into wc
            p = wc_ukn_words / wc_ukn_words.sum()
            # effective wc of unknown words (same reference as wc_unkn_words)
            wc_new_words = np.random.multinomial(max_num_new_words, p)

            # find words whose counting is non zero
            nnz = np.nonzero(wc_new_words)
            # get new words
            idx_ukn_act_new = idx_ukn_act[nnz]
            val_ukn_act_new = act[idx_ukn_act_new]
            # update new words counting
            self.lang_stats[lang]['wc'][val_ukn_act_new] += wc_new_words[nnz]
            # check if memory of some of these words can be updated
            upd_cond = self.lang_stats[lang]['wc'][val_ukn_act_new] > self.lang_stats[lang]['mem_eff'][val_ukn_act_new]
            # find indices of to-be-updated new words on act word coordinates
            idx_ukn_act_upd = idx_ukn_act_new[upd_cond]
            if idx_ukn_act_upd.size:
                new_words_idxs = idx_ukn_act_upd

        return new_words_idxs

    def learn_unknown_words(self, lang, act, act_c, kn_act_bool):
        """
            Method to add unknown words to word counter and
            to check if corresponding word memory can be updated
            Output:
                * Numpy array with indices of words whose memory can be updated
        """
        # indices of unknown words on active coordinates
        idx_ukn_act = np.where(~kn_act_bool)[0]
        # unknown words values ( indices on global coordinates == the words themselves )
        val_ukn_act = act[idx_ukn_act]
        # update counting of recognised words
        self.lang_stats[lang]['wc'][val_ukn_act] += act_c[idx_ukn_act]
        # check if memory can be updated
        upd_cond = self.lang_stats[lang]['wc'][val_ukn_act] >= self.lang_stats[lang]['mem_eff'][val_ukn_act]
        idx_ukn_act_upd = idx_ukn_act[upd_cond]
        if idx_ukn_act_upd.size:
            new_words_idxs = idx_ukn_act_upd
            return new_words_idxs

    def update_words_memory(self, lang, act, act_c, ds_factor=0.25, pct_threshold=0.9,
                            a=7.6, b=0.023, c=-0.031, d=-0.2):
        """
            Method to compute the update of word memory based on key parameters:
                S : stability
                R : retention
                t : elapsed time from last activation
            Description of MEMORY MODEL: https://www.supermemo.com/articles/stability.htm
            Assumptions ( see "HOW MANY WORDS DO WE KNOW ???" By Marc Brysbaert,
            Michaël Stevens, Paweł Mandera and Emmanuel Keuleers):
                * ~ 16000 spoken tokens per day + 16000 heard tokens per day + TV, RADIO
                * 1min reading -> 220-300 tokens with large individual differences, thus
                  in 1 h we get ~ 16000 words

            Args:
                * lang: string. Label to identify lang that to-be-updated words belong to
                * act: numpy array of integers. words whose memory will be updated
                * act_c: numpy array of integers. Counts of each word
                * ds_factor: float <= 1. Defines increase of mem stability due to passive rehearsal
                    as a fraction of that due to active rehearsal
                * pct_threshold: positive float < 1. Value to define lang knowledge in percentage.
                    If retention R for a given word is higher than pct_threshold,
                    the word is considered as well known. Otherwise, it is not
                * a, b, c, d: float parameters to define memory function from SUPERMEMO by Piotr A. Wozniak
            Output:
                * Updated word retention(R), stability(S) and last-activation time-counter(t) arrays
        """

        # compute increase in memory stability S due to (re)activation
        # TODO : I think it should be dS[reading]  < dS[media_listening]  < dS[listen_in_act_conv] < dS[speaking]

        S_act = self.lang_stats[lang]['S'][act]
        b_exp = -b
        S_act_b = numba_comp_exp(S_act, b_exp)
        R_act = self.lang_stats[lang]['R'][act]
        delta_S = numba_comp_delta_S(ds_factor, act_c, S_act_b, R_act, a, c, d)
        # update memory stability value
        self.lang_stats[lang]['S'][act] += delta_S
        # discount counts by one unit
        act_c_c = np.array(act_c)
        act_c_c -= 1
        # Simplification with good approx : we apply delta_S without iteration !!
        S_act = self.lang_stats[lang]['S'][act]
        S_act_b = numba_comp_exp(S_act, b_exp)
        R_act = self.lang_stats[lang]['R'][act]
        delta_S = numba_comp_delta_S(ds_factor, act_c_c, S_act_b, R_act, a, c, d)
        # update
        self.lang_stats[lang]['S'][act] += delta_S
        # update daily boolean mask to update elapsed-steps array t
        self.step_mask[lang][act] = True
        # set last activation time counter to zero if word act
        self.lang_stats[lang]['t'][self.step_mask[lang]] = 0.
        # compute new memory retention R and current lang_knowledge from t, S values
        self.update_memory_retention(lang, pct_threshold=pct_threshold)

    def update_memory_retention(self, lang, pct_threshold=0.9):
        # compute new memory retention R using updated t values
        fk = -self.k
        elapsed_time = self.lang_stats[lang]['t']
        lang_stability = self.lang_stats[lang]['S']
        # compute new memory retention R using updated t values
        self.lang_stats[lang]['R'] = numba_comp_R(fk, elapsed_time, lang_stability)
        # set current lang knowledge
        # update current language knowledge in percentage
        self.update_lang_pct_knowledge(lang, pct_threshold=pct_threshold)

    def update_lang_pct_knowledge(self, lang, pct_threshold=0.9):
        """ Update value that measures language knowledge in percentage
            Args:
                * lang: string. Language whose knowledge level has to be updated
                * pct_threshold: float. Value below which words are not considered fully known
            Output:
                * Updates value of self.lang_stats[lang]['pct'] array for agent's current age
        """
        # compute language knowledge in percentage
        if lang in ['L1', 'L12']:
            lang1, lang2 = ['L1', 'L12']
        else:
            lang1, lang2 = ['L2', 'L21']
        r_lang1 = self.lang_stats[lang1]['R']
        r_lang2 = self.lang_stats[lang2]['R']
        real_lang_knowledge = numba_comp_max(r_lang1, r_lang2)
        counts = numba_counter(real_lang_knowledge, pct_threshold)
        pct_value = counts / len(self.model.cdf_data['s'][self.info['age']])
        self.lang_stats[lang1]['pct'][self.info['age']] = pct_value
        # TODO: if pct < 0.1 => update language spoken to known people, since self is now really MONOLINGUAL

    def update_word_activation_elapsed_time(self, lang):
        # update last-time word use vector
        self.lang_stats[lang]['t'][~self.step_mask[lang]] += 1

    def reset_step_mask(self, lang):
        # reset agent's step mask
        self.step_mask[lang] = np.zeros(self.model.vocab_red, dtype=bool)

    def register_to_school(self, max_num_studs_per_course=25):
        def reg_conds(school):
            conds = ((self.loc_info['school'][0] is not school) or
                     (self.loc_info['school'][0] is school and not self.loc_info['school'][1])
                     )
            return conds

        # find school in cluster (with needed course available) from closest to farthest
        clust_info = self.model.geo.clusters_info[self['clust']]

        schools_with_course = [school for school in clust_info['schools']
                               if school.find_course_key(self) in school.grouped_studs]

        sorted_ixs_with_course = np.argsort([pdist([self.loc_info['home'].pos, school.pos])[0]
                                             for school in schools_with_course])
        for idx_school in sorted_ixs_with_course:
            school = schools_with_course[idx_school]
            # check best school is not current one
            if (self.loc_info['school'][0] is school) and (self.loc_info['school'][1]):
                return
            # register to new school if conditions met
            if reg_conds(school):
                # if conditions met, check if num students in course is not too high
                course_key = school.find_course_key(self)
                if len(school.grouped_studs[course_key]) <= max_num_studs_per_course:
                    # break for loop in current school
                    break
        else:
            # find school in cluster (without available course) from closest to farthest
            schools_without_course = [school for school in clust_info['schools']
                                      if school.find_course_key(self) not in school.grouped_studs]
            sorted_ixs_no_course = np.argsort([pdist([self.loc_info['home'].pos, school.pos])[0]
                                               for school in schools_without_course])
            for idx_school in sorted_ixs_no_course:
                school = schools_without_course[idx_school]
                if reg_conds(school):
                    break
            else:
                # create new school
                school = self.model.geo.add_new_school(self['clust'])
        school.assign_student(self)


class SpeakerAgent(ListenerAgent):

    """ ListenerAgent class augmented with speaking-related methods """

    @staticmethod
    def get_unique_words_and_counts(word_samples):
        # get unique words and counts
        bcs = np.bincount(word_samples)
        bool_non_zero_counts = bcs > 0
        return np.where(bool_non_zero_counts)[0], bcs[bool_non_zero_counts]

    def get_num_words_per_conv(self, conv_length='M'):
        """ Computes number of words spoken per conversation for a given age
            based on a 'num_words vs age' curve
            Args:
                * conv_length: string. Describes conversation length using keys from
                    model attribute dict 'num_words_conv' ('VS', 'S', 'M', 'L')
            Output:
                * Integer (Number of tokens)
        """
        factor = self.model.conv_length_age_factor[self.info['age']]
        return max(int(self.model.num_words_conv[conv_length] * factor), 1)

    def think_vocab(self):
        """ TODO: method to think in best known language when alone """
        pass

    def study_vocab(self, lang, delta_s_factor=1, num_words=50, num_days=1):
        """
            Method to update vocabulary without conversations ( study, reading, etc... )
                Args:
                    * lang: string. Language in which agent studies ['L1', 'L2']
                    * delta_s_factor: positive float < 1. Defines increase of mem stability
                        due to passive rehearsal as a fraction of that due to active rehearsal
                    * num_words : integer. Number of words studied per study action
                    * num_days : integer. Number of days per step the study is repeated
        """
        num_words = num_words * num_days
        word_samples = randZipf(self.model.cdf_data['s'][self.info['age']], num_words)
        # get unique words and counts
        act, act_c = self.get_unique_words_and_counts(word_samples)

        studied_words = {lang: [act, act_c]}
        # TODO : not all unknown words can be learned
        # TODO : customize 'process_unknown_words' method for read words ??
        # TODO : idea learn meaning of 30% of unknown words

        self.update_lang_arrays(studied_words, mode_type='read', delta_s_factor=delta_s_factor)

    #@BaseAgent.words_per_conv_counter
    #@BaseAgent.method_calls_counter
    def pick_vocab(self, lang, num_words=None, conv_length='M', min_age_interlocs=None,
                   biling_interloc=False, num_days=10):
        """ Method that models word choice by self agent in a conversation and
            updates agent's corresponding lang arrays
            Word choice is governed by vocabulary knowledge constraints
            Args:
                * lang: integer in [0, 1] {0:'spa', 1:'cat'}
                * num_words: integer. Number of words to be uttered. If None, number of words will be
                    picked from that of a standard short or long conversation. Default None
                * conv_length: string. If num_words is not specified, it describes conversation length
                    using keys from model attribute dict 'num_words_conv' ('VS', 'S', 'M', 'L').
                * min_age_interlocs: integer. The youngest age among all interlocutors, EXPRESSED IN STEPS.
                    It is used to modulate conversation vocabulary to younger agents
                * biling_interloc : boolean. If True, speaker word choice might be mixed (code switching), since
                    he/she is certain interlocutor will understand
                * num_days : integer [1, 10]. Number of days in one 10-day-step this kind of speech is done
            Output:
                * spoken words: dict where keys are lang labels and values are lists with words spoken
                    in lang key and corresponding counts
                * Method automatically updates self agent lang arrays after uttering spoken words
        """

        # TODO: VERY IMPORTANT -> Model language switch btw bilinguals, reflecting easiness of retrieval
        # TODO : model 'Grammatical foreigner talk' =>
        # TODO : how word choice is adapted by native speakers when speaking to adult learners

        # sample must come from AVAILABLE words in R (retrievability) !!!! This can be modeled in following STEPS

        # 1. First sample from lang CDF (that encapsulates all to-be-known concepts at a given age-step)
        # These are the thoughts or concepts a speaker tries to convey
        # TODO : VI BETTER IDEA. Known thoughts are determined by UNION of all words known in L1 + L12 + L21 + L2

        if not num_words:
            num_words = self.get_num_words_per_conv(conv_length) * num_days
        if min_age_interlocs:
            word_samples = randZipf(self.model.cdf_data['s'][min_age_interlocs], num_words)
        else:
            word_samples = randZipf(self.model.cdf_data['s'][self.info['age']], num_words)

        # get unique words and counts
        act, act_c = self.get_unique_words_and_counts(word_samples)
        # SLOWER BUT CLEANER APPROACH =>  act, act_c = np.unique(word_samples, return_counts=True)

        # check if conversation is btw bilinguals and therefore lang switch is possible
        # TODO : key is that most biling agents will not express surprise to lang mixing or switch
        # TODO : whereas monolinguals will. Feedback for correction
        # if biling_interloc:
        #     if lang == 'L1':
        #         # find all words among 'act' words where L2 retrievability is higher than L1 retrievability
        #         # Then fill L12 container with a % of those words ( their knowledge is known at first time of course)
        #
        #         self.lang_stats['L1']['R'][act] <= self.lang_stats['L2']['R'][act]
        #         L2_strongest = self.lang_stats['L2']['R'][act] == 1.
                #act[L2_strongest]

                # rand_info_access = np.random.rand(4, len(act))
                # mask_L1 = rand_info_access[0] <= self.lang_stats['L1']['R'][act]
                # mask_L12 = rand_info_access[1] <= self.lang_stats['L12']['R'][act]
                # mask_L21 = rand_info_access[2] <= self.lang_stats['L21']['R'][act]
                # mask_L2 = rand_info_access[3] <= self.lang_stats['L2']['R'][act]

        # 2. Given a lang, pick the variant that is most familiar to agent
        lang = 'L1' if lang == 0 else 'L2'
        # TODO : fix bug !! ['L12']['pct'] never gets updated in 'update_lang_knowledge', always zero
        if lang == 'L1':
            pct1, pct2 = self.get_langs_pcts(0), self.lang_stats['L12']['pct'][self.info['age']]
            lang = 'L1' if pct1 >= pct2 else 'L12'
        elif lang == 'L2':
            pct1, pct2 = self.get_langs_pcts(1), self.lang_stats['L21']['pct'][self.info['age']]
            lang = 'L2' if pct1 >= pct2 else 'L21'

        # 3. Then assess which sampled words-concepts can be successfully retrieved from memory
        # get mask for words successfully retrieved from memory
        num_act = len(act)
        mask_r = np.random.rand(num_act) <= self.lang_stats[lang]['R'][act]
        spoken_words = {lang: [act[mask_r], act_c[mask_r]]}
        # if there are missing words-concepts, they might be found in the other known language(s)
        # TODO : model depending on interlocutor (whether he is bilingual or not)
        # TODO : should pure language always get priority in computing random access ????
        # TODO : if word not yet heard in hybrid, set random creation at 50% if interloc is biling

        # TODO : if word is similar, go for it ( need to quantify similarity !!)

        if np.count_nonzero(mask_r) < num_act:
            if lang in ['L1', 'L12']:
                lang2 = 'L12' if lang == 'L1' else 'L1'
            elif lang in ['L2', 'L21']:
                lang2 = 'L21' if lang == 'L2' else 'L2'
            mask_R2 = np.random.rand(len(act[~mask_r])) <= self.lang_stats[lang2]['R'][act[~mask_r]]
            if act[~mask_r][mask_R2].size:
                spoken_words.update({lang2: [act[~mask_r][mask_R2], act_c[~mask_r][mask_R2]]})
            # if still missing words, look for them in last lang available
            if (act[mask_r].size + act[~mask_r][mask_R2].size) < len(act):
                lang3 = 'L2' if lang2 in ['L12', 'L1'] else 'L1'
                rem_words = act[~mask_r][~mask_R2]
                mask_R3 = np.random.rand(len(rem_words)) <= self.lang_stats[lang3]['R'][rem_words]
                if rem_words[mask_R3].size:
                    # VERY IMP: add to transition language instead of 'pure' one.
                    # This is the process of creation/adaption/translation
                    tr_lang = max([lang, lang2], key=len)
                    spoken_words.update({tr_lang: [rem_words[mask_R3], act_c[~mask_r][~mask_R2][mask_R3]]})
        # update speaker's lang arrays
        self.update_lang_arrays(spoken_words, delta_s_factor=1, num_days=num_days)

        return spoken_words

    def update_acquaintances(self, other, lang):
        """
            Method adds edges to known_people network when meeting (speaking)
            for the first time
            Args:
                * other: agent instance.
                * lang: integer in [0, 1]. 0 is 'L1', 1 is 'L2'
            Output:
                * method sets edge connection in 'known_people_network' between
                'self' and 'other' agents
        """
        if other not in self.model.nws.known_people_network[self]:
            self.model.nws.known_people_network.add_edge(self, other)
            self.model.nws.known_people_network[self][other].update({'num_meet': 1, 'lang': lang})
        elif (other not in self.model.nws.family_network[self] and
              other not in self.model.nws.friendship_network[self]):
            # keep track for potential friendship definition
            self.model.nws.known_people_network[self][other]['num_meet'] += 1

    def speak_in_random_subgroups(self, group, num_days=2, max_group_size=6):
        """
            Randomly select a subset of a given group
            and speak to it. Then select a random member of the subgroup
            and speak to that member.
            Args:
                * group: a list, a tuple or a set of agent instances
                * num_days: integer. Out of 10, number of days action will take place
                * max_group_size: integer > 0. Maximum size of random group
        """
        if group:
            num_mates = random.randint(1, min(len(group), max_group_size))
            # Choose n = 'num_mates' unique random mates
            # (random.sample requires a sequence since Python 3.11, so coerce
            #  the group -- which may be a set -- to a list first)
            speak_group = random.sample(list(group), num_mates)
            self.model.run_conversation(self, speak_group, num_days=num_days)
            # talk to a single random mate
            random_mate = np.random.choice(speak_group)
            self.model.run_conversation(self, random_mate, num_days=num_days)

    def update_lang_exclusion(self):
        """ Method to update exclusion counter after failed conversation
            Method is called after running 'run_conversation' model method """
        excl_lang = 'L1' if self.info['language'] == 2 else 'L2'
        self.lang_stats[excl_lang]['excl_c'][self.info['age']] += 1

    def evaluate_lang_exclusion(self, mem_window_length=5):
        """
            Method evaluates degree of linguistic exclusion during recent steps by
            computing exponential moving average of the exclusion history
            by using weights available in agent info. If EMA is higher than one,
            'react_to_lang_exclusion' method to study/learn needed language is called
            Method to be checked at the end of each step if agent is not bilingual
            Args:
                * mem_window_length: integer. Number of steps to compute EMA (length of exclusion memory)
        """
        lang = 'L2' if self.info['language'] == 0 else 'L1'
        age = self.info['age']
        # implementing exponential moving average
        excl_history = self.lang_stats[lang]['excl_c'][age - mem_window_length:age]
        excl_intensity = np.dot(self.info['excl_weights'], excl_history)
        if excl_intensity >= 1:
            self.react_to_lang_exclusion(lang)

    def react_to_lang_exclusion(self, lang):
        """
            Meanings of unknown basic words are explained to agent
            Args:
                * lang: integer in [0, 1]
        """
        # TODO: compare current vocabulary to theoretical vocab at agent age

        # If exclusion language is taught at school, teacher will communicate words meaning
        school, course_key = self.get_school_and_course()
        if lang in school.info['lang_policy']:
            # TODO : should be a new method 'teach'
            teacher = school[course_key]['teacher']
            taught_words = teacher.pick_vocab(lang, num_words=1,
                                              min_age_interlocs=self.info['age'])
            self.update_lang_arrays(taught_words, mode_type='listen', learning=True)
        else:
            # check if mates at school can teach some words 50% of steps
            pass

    def pick_friend_candidate(self):
        pass

    def check_friend_conds(self, other, min_times=None, init_max_num_friends=None):
        """
            Method to check if given agent meets compatibility conditions to become a friend
            Input:
                * other: agent instance. Instance whose friendship conditions must be checked
                * min_times: int. Minimum number of times agents need to have met
            Output:
                * Boolean. Returns True if friendship is possible, None otherwise
        """
        if not init_max_num_friends:
            max_num_friends = self.info['max_num_friends']
            max_num_friends_other = other.info['max_num_friends']
        else:
            max_num_friends = min(self.info['max_num_friends'], init_max_num_friends)
            max_num_friends_other = min(other.info['max_num_friends'], init_max_num_friends)

        if (abs(other.info['language'] - self.info['language']) <= 1 and
                len(self.model.nws.friendship_network[self]) < max_num_friends and
                len(self.model.nws.friendship_network[other]) < max_num_friends_other and
                other not in self.model.nws.friendship_network[self] and
                other not in self.model.nws.family_network[self]):
            if not min_times:
                return True
            else:
                num_times = self.model.nws.known_people_network[self][other]['num_meet']
                if num_times >= min_times:
                    comm_lang = self.model.nws.known_people_network[self][other]['lang']
                    comm_lang = 'L1' if comm_lang == 0 else 'L2'
                    age_1, age_2 = self.info['age'], other.info['age']
                    pct_1, pct_2 = (self.lang_stats[comm_lang]['pct'][age_1],
                                    other.lang_stats[comm_lang]['pct'][age_2])
                    if abs(pct_1 - pct_2) <= 0.3:
                        return True

    def make_friend(self, other):
        """ Method to implement friendship bounds between self agent and other """
        friends = [self, other]
        # who speaks first may determine communication lang
        random.shuffle(friends)
        lang = self.model.get_conv_params(friends)['lang_group']
        self.model.nws.friendship_network.add_edge(self, other, lang=lang,
                                                   weight=np.random.randint(1, 10))
        # known people network is directed graph !
        self.model.nws.known_people_network.add_edge(self, other, friends=True, lang=lang)
        self.model.nws.known_people_network.add_edge(other, self, friends=True, lang=lang)

    def stage_1(self, num_days=10):
        ags_at_home = self.loc_info['home'].agents_in.difference({self})
        ags_at_home = [ag for ag in ags_at_home if isinstance(ag, SpeakerAgent)]
        if ags_at_home:
            self.model.run_conversation(self, ags_at_home, num_days=num_days)
            for ag in np.random.choice(ags_at_home, size=min(2, len(ags_at_home)), replace=False):
                self.model.run_conversation(self, ag, num_days=num_days)


class SchoolAgent(SpeakerAgent):
    """ SpeakerAgent augmented with methods related to school activity """

    def set_educ_center_info(self, school, course_key):
        """ Convenience method to set values to school information attribute
            * args: school, course_key
        """
        self.loc_info['school'] = [school, course_key]

    def go_to_school(self):
        school = self.loc_info['school'][0]
        self.model.grid.move_agent(self, school.pos)
        school.agents_in.add(self)

    def get_school_and_course(self):
        """
            Method to get school center and corresponding course for student agent
        """

        educ_center, course_key = self.loc_info['school']
        return educ_center, course_key

    def speak_at_school(self, educ_center, course_key, num_days=7):
        """
            Method to talk with school mates and friends in an educational center,
            such as school or university
            Args:
                * educ_center: instance of agent's educational center
                * course_key: integer.
                * num_days: integer.
        """
        # TODO : filter by correlation in language preference

        speak_ags_in_course = set([ag for ag in educ_center[course_key]['students']
                                   if isinstance(ag, SchoolAgent)])
        # talk to random mates in group
        mates = speak_ags_in_course.difference({self})
        if not mates:
            # find closest course to course_key
            def f_sort(k):
                return abs(k-course_key) if k != course_key else 20
            mates = educ_center[min(educ_center.grouped_studs.keys(),
                                key=f_sort)]['students']
            mates = [ag for ag in mates if isinstance(ag, SchoolAgent)]
        self.speak_in_random_subgroups(mates, num_days=num_days)
        # talk to friends
        for friend in self.model.nws.friendship_network[self]:
            if friend in educ_center[course_key]['students']:
                self.model.run_conversation(self, friend, num_days=num_days)


class IndepAgent(SpeakerAgent):
    """ ListenerAgent class augmented with methods that enable agent
        to take independent actions """

    def move_random(self):
        """ Take a random step into any surrounding cell
            All eight surrounding cells are available as choices
            Current cell is not an output choice

            Returns:
                * modifies self.pos attribute
        """
        x, y = self.pos  # agent attr 'pos' is defined when adding agent to GRID
        possible_steps = self.model.grid.get_neighborhood((x, y),
                                                          moore=True,
                                                          include_center=False)
        chosen_cell = random.choice(possible_steps)
        self.model.grid.move_agent(self, chosen_cell)

    def move_to(self, to_loc, from_loc=None):
        """ Move to a given location (from another optionally given location) """
        pass
        #self.model.grid.move_agent(self, school.pos)

    #@BaseAgent.conv_counter
    def speak_to_group(self, group, lang=None, conv_length='M',
                       biling_interloc=False, num_days=10):
        """
            Make self agent speak to a group of listening people
            Args:
                * group: list of agent instances
                * lang: integer
                * conv_length: string. Conversation length
            Output:
                *  Updated lang arrays for both self agent and listening group
        """

        min_age_interlocs = min([ag.info['age'] for ag in group])
        # TODO : use 'get_conv_params' to get the parameters ???

        if lang is None:
            lang = self.model.get_conv_params([self] + group)['lang_group'][0]

        spoken_words = self.pick_vocab(lang, conv_length=conv_length, min_age_interlocs=min_age_interlocs,
                                       biling_interloc=biling_interloc, num_days=num_days)
        # update listeners' lang arrays
        for ag in group:
            ag.update_lang_arrays(spoken_words, mode_type='listen', delta_s_factor=0.1, num_days=num_days)

    def pick_random_friend(self, ix_agent):
        """
            Method selects a row of friendship adjacent matrix based on agent index
            It then picks a random friend based on probabilities of friendship intensity

            Args:
                * ix_agent: integer. agent index in schedule agent list
        """

        # get current agent neighboring nodes ids
        adj_mat = self.model.nws.adj_mat_friend_nw[ix_agent]
        # get agents ids and probs
        if adj_mat.data.size:
            ix = np.random.choice(adj_mat.indices, p=adj_mat.data)
            picked_agent = self.model.schedule.agents[ix]
            return picked_agent

    def speak_to_random_friend(self, ix_agent, num_days):
        """ Method to speak to a randomly chosen friend
            Args:
                * ix_agent: integer. Agent index in schedule agent list
                * num_days: integer. Number of days out of ten when action takes place on average
        """
        random_friend = self.pick_random_friend(ix_agent)
        if random_friend:
            conv_length = random.choice(['M', 'L'])
            self.model.run_conversation(self, random_friend,
                                        def_conv_length=conv_length, num_days=num_days)

    def react_to_lang_exclusion(self, lang, mem_window_length=5):
        """
            Method to model reaction to linguistic exclusion. Agent
            will try to learn language he does not know
            Args:
                * lang: string. Language ('L1' or 'L2')
                * mem_window_length: integer. Measures maximum number of steps backwards
                    that exclusion is 'remembered'
        """
        self.study_vocab(lang)
        # TODO : study vocabulary
        # TODO : ask friends to teach unknown words
        # TODO : join conversations in target language
        # TODO : listen to media

    def meet_agent(self):
        pass
    # TODO : method to meet new agents

    def check(self):
        """ Check num_meet in known people network to filter candidates """

        # TODO: pick random acquaintance with condition num meets > 5

        # TODO : check similar age
        # TODO : check lang profile not too distant
        # TODO : check residence in same cluster or  close one
        pass


class Baby(ListenerAgent):

    """
        Agent from 0 to 2 years old.
        It must be initialized only from 'reproduce' method in Young and Adult agent classes
        Args:
            * father, mother : class instances
            * lang_with_father, lang_with_mother: integer [0, 1]. Defines lang of agent with father, mother
    """
    age_low, age_high = 0, 2

    def __init__(self, father, mother, lang_with_father, lang_with_mother, *args, school=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.model.nws.set_newborn_family_links(self, father, mother, lang_with_father, lang_with_mother)
        if school:
            school.assign_student(self)
        else:
            self.set_educ_center_info(school, None)

    def set_educ_center_info(self, school, course_key):
        self.loc_info['school'] = [school, course_key]

    def get_school_and_course(self):
        """
            Method to get school center and corresponding course for student agent
        """
        educ_center, course_key = self.loc_info['school']
        return educ_center, course_key

    def stage_1(self, num_days=10):
        # listen to close family at home
        # all activities are DRIVEN by current agent
        if self.info['age'] > self.model.steps_per_year:
            ags_at_home = [ag for ag in self.loc_info['home'].agents_in.difference({self})
                           if isinstance(ag, SpeakerAgent)]
            for ag in ags_at_home:
                self.listen(to_agent=ag, min_age_interlocs=self.info['age'], num_days=num_days)
            if not self.loc_info['school'][1]:
                self.register_to_school()

    def stage_2(self, num_days=7, prob_father=0.2):
        # go to daycare with mom or dad - 7 days out of 10
        # ONLY 7 out of 10 days are driven by current agent (rest by parents or caretakers)
        if self.info['age'] > self.model.steps_per_year:
            # move self to school and identify teacher
            school, course_key = self.get_school_and_course()
            self.model.grid.move_agent(self, school.pos)
            school.agents_in.add(self)
            teacher = school.grouped_studs[course_key]['teacher']
            # check if school parent is available
            school_parent = 'mother' if random.random() > prob_father else 'father'
            school_parent = self.get_family_relative(school_parent)
            if school_parent:
                self.listen(to_agent=school_parent, min_age_interlocs=self.info['age'], num_days=num_days)
                self.model.run_conversation(teacher, school_parent,
                                            def_conv_length='S', num_days=int(num_days/3))
            # make self interact with teacher
            # TODO : a part of speech from teacher to all course(driven from teacher stage method)
            for _ in range(2):
                self.listen(to_agent=teacher, min_age_interlocs=self.info['age'],
                            num_days=num_days)
        # model week-ends time are modeled in PARENTS stages

    def stage_3(self, num_days=7, prob_father=0.2):
        # parent comes to pick up and speaks with other parents. Then baby listens to parent on way back
        if self.info['age'] > self.model.steps_per_year:
            # continue school day
            school, course_key = self.get_school_and_course()
            teacher = school.grouped_studs[course_key]['teacher']
            for _ in range(2):
                self.listen(to_agent=teacher, min_age_interlocs=self.info['age'],
                            num_days=num_days)
            # check if school parent is available
            school_parent = 'mother' if random.random() > prob_father else 'father'
            school_parent = self.get_family_relative(school_parent)
            if school_parent:
                self.model.grid.move_agent(school_parent, school.pos)
                self.listen(to_agent=school_parent, min_age_interlocs=self.info['age'], num_days=num_days)
            parents = [ag for ag in self.model.grid.get_cell_list_contents(school.pos)
                       if isinstance(ag, Young)]
            if parents and school_parent:
                num_peop = random.randint(1, min(len(parents), 4))
                self.model.run_conversation(school_parent, random.sample(parents, num_peop))
            school.remove_agent_in(self)

        # TODO : pick random friends from parents. Set up meeting with them

    def stage_4(self):
        # baby goes to bed early during week
        self.go_back_home()
        self.stage_1(num_days=3)
        if self.info['age'] == self.age_high * self.model.steps_per_year:
            self.evolve(Child)


class Child(SchoolAgent):

    age_low, age_high = 2, 12

    def __init__(self, *args, school=None, **kwargs):
        # TODO: add extra args specific to this class if needed
        super().__init__(*args, **kwargs)
        if school:
            school.assign_student(self)
        else:
            self.set_educ_center_info(school, None)

    def speak_at_school(self, school, course_key, num_days=7):
        super().speak_at_school(school, course_key, num_days=num_days)
        # talk with teacher
        teacher = school[course_key]['teacher']
        self.model.run_conversation(teacher, self, num_days=2)

    def stage_1(self, num_days=7):
        SpeakerAgent.stage_1(self, num_days=num_days)

    def stage_2(self, num_days=7):
        # go to daycare with mom or dad - 7 days out of 10
        # ONLY 7 out of 10 days are driven by current agent (rest by parents or caretakers)
        school, course_key = self.get_school_and_course()
        self.go_to_school()
        # check if school parent is available
        school_parent = 'mother' if random.uniform(0, 1) > 0.2 else 'father'
        school_parent = self.get_family_relative(school_parent)
        if school_parent:
            teacher = school[course_key]['teacher']
            self.model.run_conversation(self, school_parent, num_days=num_days)
            self.model.run_conversation(teacher, school_parent, num_days=2)
        # talk with school mates, friends and teacher
        # TODO : filter mate selection by correlation in language preference
        self.speak_at_school(school, course_key, num_days=num_days)
        # TODO : week-ends time are modeled in PARENTS stages

    def stage_3(self, num_days=7):
        school, course_key = self.get_school_and_course()
        self.speak_at_school(school, course_key, num_days=num_days)
        school_parent = 'mother' if random.uniform(0, 1) > 0.2 else 'father'
        school_parent = self.get_family_relative(school_parent)
        if school_parent:
            self.model.grid.move_agent(school_parent, school.pos)
            self.model.run_conversation(self, school_parent, num_days=num_days)
        parents = [ag for ag in self.model.grid.get_cell_list_contents(school.pos)
                   if isinstance(ag, Adult)]
        if parents and school_parent:
            num_peop = random.randint(1, min(len(parents), 4))
            self.model.run_conversation(school_parent, random.sample(parents, num_peop))
        school.remove_agent_in(self)

    def stage_4(self, num_days=10):
        self.go_back_home()
        self.stage_1(num_days=num_days)
        if self.info['age'] == self.age_high * self.model.steps_per_year:
            self.evolve(Adolescent)


class Adolescent(IndepAgent, SchoolAgent):

    age_low, age_high = 12, 18

    def __init__(self, *args, school=None, **kwargs):
        super().__init__(*args, **kwargs)
        if school:
            school.assign_student(self)
        else:
            self.set_educ_center_info(school, None)

    def speak_at_school(self, school, course_key, num_days=7):
        super().speak_at_school(school, course_key, num_days=num_days)
        # talk with teacher
        teacher = school[course_key]['teacher']
        self.model.run_conversation(teacher, self, num_days=1)

        age = self.info['age']
        # Define random group at school level with adjacent courses
        mates = {st for st in school.info['students']
                 if st.info['age'] in [age-1, age, age+1]}
        mates = mates.difference({self})
        self.speak_in_random_subgroups(mates, num_days=3)

    def evolve(self, new_class, ret_output=False, university=None, upd_course=False):

        grown_agent = super().evolve(new_class, ret_output=True, upd_course=upd_course)
        # new agent will not go to school in any case
        del grown_agent.loc_info['school']
        # find out growth type
        if isinstance(grown_agent, YoungUniv):
            if university:
                fac_key = random.choice(string.ascii_letters[:5])
                fac = university.faculties[fac_key]
                fac.assign_student(grown_agent, course_key=19, hire_t=False)
                # agent moves to new home if he has to change cluster to attend univ
                if fac.info['clust'] != grown_agent['clust']:
                    grown_agent.move_to_new_home()
            else:
                raise Exception('university instance must be provided to grow into a YoungUniv')
        elif isinstance(grown_agent, Young):
            grown_agent.info.update({'married': False, 'num_children': 0})
            grown_agent.loc_info['job'] = None
        if ret_output:
            return grown_agent

    def stage_1(self, ix_agent, num_days=7):
        SpeakerAgent.stage_1(self, num_days=num_days)

    def stage_2(self, ix_agent, num_days=7):
        school, course_key = self.get_school_and_course()
        self.model.grid.move_agent(self, school.pos)
        school.agents_in.add(self)
        self.speak_at_school(school, course_key, num_days=num_days)

    def stage_3(self, ix_agent, num_days=7):
        school, course_key = self.get_school_and_course()
        self.speak_at_school(school, course_key, num_days=num_days)
        school.remove_agent_in(self)
        # study
        if school.info['lang_policy'] == [0, 1]:
            self.study_vocab('L1', num_words=50, num_days=5)
            self.study_vocab('L2', num_words=25, num_days=5)
        elif school.info['lang_policy'] == [1]:
            self.study_vocab('L1', num_days=5)
            self.study_vocab('L2', num_days=5)
        elif school.info['lang_policy'] == [1, 2]:
            self.study_vocab('L1', num_words=25, num_days=5)
            self.study_vocab('L2', num_words=50, num_days=5)
        if self.model.nws.friendship_network[self]:
            # TODO : add groups, more than one friend
            self.speak_to_random_friend(ix_agent, num_days=5)

        self.listen(num_days=1)

    def stage_4(self, ix_agent, num_days=7):
        self.go_back_home()
        self.stage_1(ix_agent, num_days=num_days)
        # go out with friends at least once
        num_out = random.randint(1, 5)
        if self.model.nws.friendship_network[self]:
            # TODO : add groups, more than one friend (friends of friends)
            self.speak_to_random_friend(ix_agent, num_days=num_out)


class Young(IndepAgent):

    age_low, age_high = 19, 30

    def __init__(self, *args, married=False, num_children=0, job=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.info['married'] = married
        self.info['num_children'] = num_children
        self._set_up_init_job(job)

    def _set_up_init_job(self, job):
        self.loc_info['job'] = None
        if job:
            job.hire_employee(self)

    def check_partner(self, cand_agent, max_age_diff=10, thresh_comm_lang=0.3):
        """ Check conditions that must be satisfied in order to get married """

        # measure max age difference in steps
        max_age_diff = self.model.steps_per_year * max_age_diff

        # define boolean variables
        link_description = self.model.nws.known_people_network[self][cand_agent]
        sex_diff = cand_agent.info['sex'] != self.info['sex']
        lang_diff = abs(cand_agent.info['language'] - self.info['language'])
        age_diff = abs(self.info['age'] - cand_agent.info['age'])

        # define knowledge of each agent in common language
        common_lang = link_description['lang']
        common_lang = 'L1' if common_lang == 0 else 'L2'
        pct_1 = self.lang_stats[common_lang]['pct'][self.info['age']]
        pct_2 = cand_agent.lang_stats[common_lang]['pct'][cand_agent.info['age']]

        # define 4 conditions to marry
        num_meetings_condition = 'num_meet' in link_description and link_description['num_meet'] > 10
        diffs_condition = sex_diff and age_diff <= max_age_diff and lang_diff <= 1
        pre_conditions = isinstance(cand_agent, Young) and not cand_agent.info['married']
        lang_threshold_condition = pct_1 > thresh_comm_lang and pct_2 > thresh_comm_lang

        if num_meetings_condition and diffs_condition and pre_conditions and lang_threshold_condition:
            return True

    def look_for_partner(self, avg_years=6, max_age_diff=10, thresh_comm_lang=0.3):
        """ Find partner every avg_years if agent is not married yet. Marrying agents
            must have a sufficiently high knowledge of common language

             Args:
                 * avg_years: integer. Years on average to get a partner (expressed in years)
                 * max_age_diff: integer. Max difference in years between partners to get married
                 * thresh_comm_lang: float. Minimum required knowledge of common lang by each agent
        """
        # first check luck
        if random.random() < 1 / (avg_years * self.model.steps_per_year):
            # find suitable partner amongst known people
            acq_network = self.model.nws.known_people_network
            for ag in acq_network[self]:
                if 'family' not in acq_network[self][ag]:
                    if self.check_partner(ag, max_age_diff=max_age_diff,
                                          thresh_comm_lang=thresh_comm_lang):
                        self.get_married(ag)
                        break

    def get_married(self, ag):
        """ Args:
                * ag: agent instance. Agent to marry to
        """
        # set marriage flags and links between partners
        self.info['married'] = True
        ag.info['married'] = True
        fam_nw = self.model.nws.family_network
        lang = self.model.nws.known_people_network[self][ag]['lang']
        # family network is directed Graph !!
        fam_nw.add_edge(self, ag, lang=lang, fam_link='consort')
        fam_nw.add_edge(ag, self, lang=lang, fam_link='consort')
        # find appartment to move in together
        self.move_to_new_home()

    def reproduce(self, day_prob=0.0014, max_num_children=4):
        """ Method to give birth to a new agent if conditions and likelihoods are met """

        # TODO: check method integration with creation of 'Baby' class
        if (random.random() < day_prob and
            self.info['num_children'] < max_num_children and
            self.info['married']):
            id_baby = self.model.set_available_ids.pop()
            # get consort
            consort = self.get_family_relative('consort')
            # find out baby language attribute and langs with parents
            newborn_lang, lang_with_father, lang_with_mother = self.model.get_newborn_lang(self, consort)
            # Determine baby's sex
            sex = 'M' if random.random() > 0.5 else 'F'
            father, mother = (self, consort) if self.info['sex'] == 'M' else (consort, self)
            # Create baby instance
            baby = Baby(father, mother, lang_with_father, lang_with_mother,
                        self.model, id_baby, newborn_lang, sex, home=self.loc_info['home'])
            # Add agent to grid, schedule, network and clusters info
            self.model.add_new_agent_to_model(baby)
            # Update num of children for both self and consort
            self.info['num_children'] += 1
            consort.info['num_children'] += 1

    def pick_cluster_for_job_search(self, keep_cluster=False):

        """
            Method to select a cluster where to look for a job and get hired.
            Args:
                * keep_cluster: boolean. If True, looks for job only in current cluster.
                    It defaults to False
        """

        # TODO: make model that takes majority language of clusters into account
        # TODO: create weights for choice that depend on distance and language
        # get current agent cluster
        clust = self['clust']
        if not keep_cluster:
            # look for job on any cluster with bias towards the current one
            # make array of cluster indexes with equal weight equal to one
            clust_ixs = np.ones(self.model.num_clusters)
            # assign a higher weight to current cluster so that it is picked more than half the times
            clust_ixs[clust] = self.model.num_clusters
            # define probabilities to pick cluster
            clust_probs = clust_ixs / clust_ixs.sum()
            # pick a random cluster according to probabs
            job_clust = np.random.choice(np.array(self.model.num_clusters), p=clust_probs)
        else:
            # look for job on current cluster only
            job_clust = clust

        return job_clust

    def get_job(self, keep_cluster=False, move_home=True, ignore_lang_constraint=False):
        """
            Assign agent to a random job unless either personal or linguistic constraints
            do not make it possible
            Args:
                * keep_cluster: boolean. If True, job search will be limited to agent's current cluster
                    Otherwise, all clusters might be searched. It defaults to False
                * move_home: boolean. True if moving to a new home is allowed
                * ignore_lang_constraint: boolean. If True, lang knowledge is ignored when
                    hiring. Default False
            Output:
                * If constraints allow it, method assigns a new job to agent
        """
        # TODO: break while loop to avoid infinite looping

        job_clust = self.pick_cluster_for_job_search(keep_cluster=keep_cluster)
        # pick a job from chosen cluster
        job = np.random.choice(self.model.geo.clusters_info[job_clust]['jobs'])
        if job.num_places and job.check_cand_conds(self, keep_cluster=keep_cluster):
            job.hire_employee(self, move_home=move_home,
                              ignore_lang_constraint=ignore_lang_constraint)

    def get_current_job(self):
        """ Method that returns agent's current job """
        try:
            job = self.loc_info['job']
        except KeyError:
            job = None
        return job

    def move_to_new_home(self, marriage=True):
        """
            Method to move self agent and other optional agents to a new home
            Args:
                * marriage: boolean. Specifies if moving is because of marriage or not. If False,
                    it is assumed moving is because of job reasons ( 'self' agent has a new job)
            Output:
                * 'self' agent is assigned a new home together with his/her family.
                    If 'self' agent is married, partner will also try to find a new job.
                    Children, if any,  will be assigned a new school in cluster of parent's new job
        """

        if marriage:
            # TODO: pensioners can marry but have no jobs !!!
            # self already has a job since it is a pre-condition to move to a new home
            # get 'self' agent job and cluster
            job_1 = self.get_current_job()
            clust_1 = self['clust']
            # get consort job and cluster
            consort = self.get_family_relative('consort')
            clust_2 = consort['clust']
            job_2 = consort.get_current_job()
            # check if consort has a job
            if job_2:
                if clust_1 == clust_2:
                    # both consorts keep jobs
                    job_change = None
                    new_home = self.find_home(criteria='half_way')
                else:
                    # married agents live and work in different clusters
                    # move to the town with more job offers
                    num_jobs_1 = len(self.model.geo.clusters_info[clust_1]['jobs'])
                    num_jobs_2 = len(consort.model.geo.clusters_info[clust_2]['jobs'])
                    if num_jobs_1 >= num_jobs_2:
                        # consort gives up job_2
                        job_change = 'consort'
                        new_home = self.find_home()
                        job_2.remove_employee(consort)
                    else:
                        # self gives up job_1
                        job_change = 'self'
                        new_home = consort.find_home()
                        job_1.remove_employee(self)
            else:
                # consort has no job
                job_change = 'consort'
                new_home = self.find_home()

            # assign new home
            new_home.assign_to_agent([self, consort])
            # update job status for agent that must switch
            if job_change == 'consort':
                consort.get_job(keep_cluster=True)
            elif job_change == 'self':
                self.get_job(keep_cluster=True)
        else:
            # moving for job reasons -> family, if any, will follow
            # new job already assigned to self
            # find close home in new job cluster
            new_home = self.find_home()
            if not self.info['married']:
                moving_agents = [self]
                new_home.assign_to_agent(moving_agents)
            else:
                # partner has to find job in new cluster
                consort = self.get_family_relative('consort')
                moving_agents = [self, consort]
                new_home.assign_to_agent(moving_agents)
                job_2 = consort.get_current_job()
                if job_2:
                    job_2.remove_employee(consort)
                    # consort looks for job in current cluster while keeping new home
                    consort.get_job(keep_cluster=True, move_home=False)
            # find out if there are children that will have to move too
            children = self.get_family_relative('child')
            children = [child for child in children
                        if not isinstance(child, (Young, YoungUniv))]
            new_home.assign_to_agent(children)
            # remove children from old school, enroll them in a new one
            for child in children:
                if child.info['age'] > self.model.steps_per_year:
                    child.register_to_school()

    def find_home(self, criteria='close_to_job'):
        """
            Method to find an empty home in same cluster as that of agent's job
            Args:
                * criteria: string. Defines criteria to find home. Options are:
                    1) 'close_to_job' : finds random house relatively close
                    to self agent job, regardless of consort's job location
                    2) 'half_way'. Finds random house in job cluster half way
                    between self agent's job and consort's job, if they both live
                    in same cluster
            Output:
                * Method returns a Home class instance that meets demanded criteria
        """

        # get cluster of agent's job
        job_1 = self.get_current_job()
        job_1_clust = job_1.info['clust']
        # get all free homes from job cluster
        free_homes_job_1_clust = [home for home in self.model.geo.clusters_info[job_1_clust]['homes']
                                  if not home.info['occupants']]
        if free_homes_job_1_clust:
            if criteria == 'close_to_job':
                # Sort all available homes by provided criteria
                sorted_homes = sorted(free_homes_job_1_clust,
                                      key=lambda home: pdist([job_1.pos, home.pos])[0])
                # pick random home from the closest half to new job
                home_ix = random.randint(0, min(ceil(len(sorted_homes) / 2), len(sorted_homes) - 1))
                new_home = sorted_homes[home_ix]
            elif criteria == 'half_way':
                consort = self.get_family_relative('consort')
                job_2 = consort.get_current_job()
                # define sum of distances from jobs to each home to sort by it
                job_dist_fun = lambda home: (pdist([job_1.pos, home.pos]) + pdist([job_2.pos, home.pos]))[0]
                sorted_homes = sorted(free_homes_job_1_clust, key=job_dist_fun)
                new_home = sorted_homes[0]
            elif criteria == 'empty':
                new_home = random.choice(free_homes_job_1_clust)
        else:
            # build new home
            new_home = self.model.geo.add_new_home(job_1_clust)

        return new_home

    def go_to_job(self):
        """
            Method to move agent to job cell coordinates and add agent to
            agents currently in office
        """
        job = self.loc_info['job']
        if self not in job.agents_in:
            self.model.grid.move_agent(self, self.loc_info['job'].pos)
            job.agents_in.add(self)
            self.info['job_steps'] += 1

    def gather_family(self):
        """
            Meet brothers or sisters with children
            one time per step with 50% chance
        """
        pass

    def gather_friends(self):
        pass

    def speak_to_customer(self, num_days=5):
        job = self.loc_info['job']
        try:
            rand_cust_job = random.choice(list(self.model.nws.jobs_network[job]))
            rand_customer = random.choice(list(rand_cust_job.info['employees']))
            self.model.run_conversation(self, rand_customer, num_days=num_days)
        except IndexError:
            pass

    def stage_1(self, ix_agent, num_days=7):
        self.evaluate_lang_exclusion()
        SpeakerAgent.stage_1(self, num_days=num_days)

    def stage_2(self, ix_agent):
        if not self.loc_info['job']:
            self.get_job()
        else:
            self.go_to_job()
            job = self.loc_info['job']
            self.speak_to_customer(num_days=3)
            colleagues = job.info['employees']
            self.speak_in_random_subgroups(colleagues, num_days=7)

    def stage_3(self, ix_agent):
        if not self.loc_info['job']:
            self.get_job()
        else:
            job = self.loc_info['job']
            colleagues = job.info['employees']
            self.speak_in_random_subgroups(colleagues, num_days=7)
            self.speak_to_customer(num_days=3)
        self.listen(num_days=1)

    def stage_4(self, ix_agent):
        self.go_back_home()
        self.stage_1(ix_agent, num_days=7)
        if self.model.nws.friendship_network[self]:
            self.speak_to_random_friend(ix_agent, num_days=3)
        if not self.info['married'] and self.loc_info['job']:
            self.look_for_partner()
        if self.info['age'] == self.age_high * self.model.steps_per_year:
            self.evolve(Adult)


class YoungUniv(Adolescent):
    age_low, age_high = 18, 24

    def __init__(self, *args, university=None, fac_key=None, **kwargs):
        BaseAgent.__init__(self, *args, **kwargs)

        if university and fac_key:
            university[fac_key].assign_student(self)
        elif university and not fac_key:
            fac_key = random.choice(string.ascii_letters[:5])
            university[fac_key].assign_student(self)
        else:
            self.set_educ_center_info(fac=None, course_key=None)

    def set_educ_center_info(self, fac, course_key):
        if fac:
            self.loc_info['university'] = [fac.univ, course_key, fac.info['type']]
        else:
            self.loc_info['university'] = [None, course_key, None]

    def get_school_and_course(self):
        educ_center, course_key, fac_key = self.loc_info['university']
        educ_center = educ_center[fac_key]
        return educ_center, course_key

    def evolve(self, new_class, ret_output=False, upd_course=False):
        grown_agent = BaseAgent.evolve(self, new_class, ret_output=True,
                                       upd_course=upd_course)
        # new agent will not go to university in any case
        del grown_agent.loc_info['university']
        grown_agent.info.update({'married': False, 'num_children': 0})
        grown_agent.loc_info['job'] = None
        if ret_output:
            return grown_agent

    def move_to_new_home(self, max_num_occup=6):
        """
            Method to move self agent and other optional agents to a new home close to university
            Args:
        """
        # move close to university
        univ = self.loc_info['university'][0]
        clust_univ = univ.info['clust']
        # make list of either empty homes or homes occupied only by YoungUniv in univ cluster
        univ_homes = [home for home in self.model.geo.clusters_info[clust_univ]['homes']
                      if not home.info['occupants'] or
                      all([isinstance(x, YoungUniv) for x in home.info['occupants']])]
        # define function to measure distance home-univ
        univ_dist_fun = lambda home: pdist([univ.pos, home.pos])[0]
        # sort homes by distance to univ, pick first that has less than max_num_occup occupants
        sorted_homes = sorted(univ_homes, key=univ_dist_fun)
        for h in sorted_homes:
            if len(h.info['occupants']) < max_num_occup:
                new_home = h
                break
        else:
            # no home available
            new_home = self.model.geo.add_new_home(clust_univ)
        # assign new home to agent
        new_home.assign_to_agent(self)

    def speak_at_school(self, faculty, course_key, num_days=7):
        super().speak_at_school(faculty, course_key, num_days=num_days)
        age = self.info['age']
        # Speak to random group at university level with adjacent courses
        univ = self.loc_info['university'][0]
        mates = {st for st in univ.info['students']
                 if st.info['age'] in [age-1, age, age+1]}
        mates = mates.difference({self})
        self.speak_in_random_subgroups(mates, num_days=2)

    def stage_1(self, ix_agent, num_days=7):
        self.evaluate_lang_exclusion()
        SpeakerAgent.stage_1(self, num_days=num_days)

    def stage_2(self, ix_agent, num_days=7):
        faculty, course_key = self.get_school_and_course()
        self.speak_at_school(faculty, course_key, num_days=num_days)

    def stage_3(self, ix_agent, num_days=7):
        faculty, course_key = self.get_school_and_course()
        self.speak_at_school(faculty, course_key, num_days=num_days)

    def stage_4(self, ix_agent):
        self.go_back_home()
        super().stage_4(ix_agent)
        # TODO: visit family, meet siblings


class Adult(Young): # from 30 to 65

    age_low, age_high = 30, 65

    def evolve(self, new_class, ret_output=False):
        grown_agent = super().evolve(new_class, ret_output=True)
        # new agent will not have a job if Pensioner
        if not isinstance(grown_agent, Teacher):
            try:
                del grown_agent.loc_info['job']
            except KeyError:
                pass
        if ret_output:
            return grown_agent

    def reproduce(self, day_prob=0.0024, limit_age=40):
        if self.info['age'] <= limit_age * self.model.steps_per_year:
            super().reproduce(day_prob=day_prob)

    def look_for_partner(self, avg_years=4, age_diff=10, thresh_comm_lang=0.3):
        super().look_for_partner(avg_years=avg_years)

    def gather_family(self):
        pass

    def gather_siblings(self):
        pass

    def stage_1(self, ix_agent, num_days=7):
        self.evaluate_lang_exclusion()
        SpeakerAgent.stage_1(self, num_days=num_days)

    def stage_2(self, ix_agent):
        super().stage_2(ix_agent)

    def stage_3(self, ix_agent):
        super().stage_3(ix_agent)
        # TODO: if agent has no parents, he/she must lead family gathering if possible

    def stage_4(self, ix_agent, num_days=7):
        self.go_back_home()
        self.stage_1(ix_agent, num_days=7)
        if self.model.nws.friendship_network[self]:
            self.speak_to_random_friend(ix_agent, num_days=3)
        if not self.info['married'] and self.loc_info['job']:
            self.look_for_partner()
        if self.info['age'] == self.model.steps_per_year * self.age_high:
            self.evolve(Pensioner)


class Worker(Adult):

    pass


class Teacher(Adult):

    def _set_up_init_job(self, job):
        self.loc_info['job'] = None
        if job and self.info['language'] in job.info['lang_policy']:
            job.assign_teacher(self)

    def get_school_and_course(self):
        """
            Method to get school center and corresponding course for teacher agent
        """
        educ_center, course_key = self.loc_info['job']
        return educ_center, course_key

    def get_students(self):
        school, course_key = self.get_school_and_course()
        if course_key:
            return school[course_key]['students']

    def evolve(self, new_class, ret_output=False):
        grown_agent = super().evolve(new_class, ret_output=True)
        if ret_output:
            return grown_agent

    def get_current_job(self):
        """ Method that returns agent's current job """
        try:
            job = self.loc_info['job'][0]
        except (KeyError, TypeError):
            job = None
        return job

    def get_job(self, keep_cluster=False, move_home=True):
        # get cluster where job will be found
        clust = self.pick_cluster_for_job_search(keep_cluster=keep_cluster)
        # assign teacher to first school that matches teacher lang profile
        for school in self.model.geo.clusters_info[clust]['schools']:
            if self.info['language'] in school.info['lang_policy']:
                school.assign_teacher(self)
                break

    def random_death(self, ret_out=False):
        BaseAgent.random_death(self, ret_out=True)

    def speak_to_class(self):
        # TODO: help students to learn meaning of unknown words
        job, course_key = self.loc_info['job']
        if course_key:
            # teacher speaks to entire course
            studs = list(job.grouped_studs[course_key]['students'])
            if studs:
                self.speak_to_group(studs, lang=0, conv_length='L', num_days=7)

    def speak_with_colleagues(self):
        pass

    def stage_1(self, ix_agent, num_days=7):
        super().stage_1(ix_agent, num_days=num_days)

    def stage_2(self, ix_agent):
        if not self.loc_info['job']:
            # TODO: make sure agent meets lang requirements to get job as teacher, study otherwise
            self.get_job(keep_cluster=True)
        else:
            job, course_key = self.get_school_and_course()
            self.model.grid.move_agent(self, job.pos)
            job.agents_in.add(self)
            self.speak_to_class()
            colleagues = job.info['employees']
            self.speak_in_random_subgroups(colleagues)

    def stage_3(self, ix_agent):
        if not self.loc_info['job']:
            self.get_job(keep_cluster=True)
        else:
            job, course_key = self.get_school_and_course()
            self.speak_to_class()
            colleagues = job.info['employees']
            self.speak_in_random_subgroups(colleagues)
            self.speak_to_random_friend(ix_agent, num_days=3)

    def stage_4(self, ix_agent, num_days=7):
        self.go_back_home()
        self.stage_1(ix_agent, num_days=7)
        if self.model.nws.friendship_network[self]:
            self.speak_to_random_friend(ix_agent, num_days=3)
        if not self.info['married'] and self.loc_info['job']:
            self.look_for_partner()


class TeacherUniv(Teacher):

    def get_school_and_course(self):
        """
            Method to get school center and corresponding course for teacher agent
        """
        educ_center, course_key, fac_key = self.loc_info['job']
        educ_center = educ_center.faculties[fac_key]
        return educ_center, course_key

    def get_current_job(self):
        try:
            job = self.loc_info['job']
            job = job[0][job[2]]
        except (KeyError, TypeError):
            job = None
        return job

    def get_job(self, keep_cluster=False, move_home=True):
        if keep_cluster:
            clust = self.pick_cluster_for_job_search(keep_cluster=keep_cluster)
        else:
            clust = np.random.choice(self.model.geo.get_clusters_with_univ())

        if 'university' in self.model.geo.clusters_info[clust]:
            univ = self.model.geo.clusters_info[clust]['university']
            if self.info['language'] in univ.info['lang_policy']:
                fac = univ[random.choice('abcde')]
                fac.assign_teacher(self)

    def random_death(self, ret_out=False):
        BaseAgent.random_death(self, ret_out=ret_out)

    def speak_to_class(self):
        job, course_key, fac_key = self.loc_info['job']
        if course_key:
            # teacher speaks to entire course
            studs = list(job[fac_key].grouped_studs[course_key]['students'])
            if studs:
                self.speak_to_group(studs, 0)

    def stage_2(self, ix_agent):
        if not self.loc_info['job']:
            self.get_job(keep_cluster=True)
        else:
            fac, course_key = self.get_school_and_course()
            self.model.grid.move_agent(self, fac.pos)
            fac.agents_in.add(self)
            self.speak_to_class()

            colleagues = fac.info['employees']
            self.speak_in_random_subgroups(colleagues)

    def stage_3(self, ix_agent):
        if not self.loc_info['job']:
            self.get_job(keep_cluster=True)
        else:
            fac, course_key = self.get_school_and_course()
            self.speak_to_class()
            colleagues = fac.info['employees']
            self.speak_in_random_subgroups(colleagues)

            self.speak_to_random_friend(ix_agent, num_days=3)


class Pensioner(Adult):
    """ Agent from 65 to death """

    def __init__(self, *args, married=False, num_children=0, **kwargs):
        BaseAgent.__init__(self, *args, **kwargs)
        self.info['married'] = married
        self.info['num_children'] = num_children

    def find_home(self, clust):
        """ Overrides method since pensioners have no job """
        # TODO
        pass

    def random_death(self, ret_out=False):
        if self.info['age'] <= self.max_life_steps - 2:
            BaseAgent.random_death(self, ret_out=False)
        else:
            self.model.remove_after_death(self)
            if ret_out:
                return True

    def gather_family(self, num_days=1, freq=0.1):
        if random.random() < freq:
            consort = self.get_family_relative('consort')
            consort = [consort] if consort else []
            # check which children are in same cluster
            children = self.get_family_relative('child')
            children = [child for child in children if child['clust'] == self['clust']]
            ch_consorts = [child.get_family_relative('consort') for child in children]
            ch_consorts = [c for c in ch_consorts if c]
            grandchildren = self.get_family_relative('grandchild')
            grandchildren = [gc for gc in grandchildren if type(gc) != Baby]

            all_family = children + ch_consorts + grandchildren + consort
            adults = children + ch_consorts + consort

            # make grandchildren talk to each other
            if len(grandchildren) >= 3:
                random.shuffle(grandchildren)
                self.model.run_conversation(grandchildren[0], grandchildren[1:], num_days=1)
            # speak to children
            self.model.run_conversation(self, children, num_days=1)
            # speak to children and consorts
            self.speak_in_random_subgroups(adults, num_days=1,
                                           max_group_size=len(adults))
            # speak to grandchildren
            self.speak_in_random_subgroups(grandchildren, num_days=1,
                                           max_group_size=len(grandchildren))
            # speak to all family
            self.model.run_conversation(self, all_family, num_days=1)

    def speak_to_children(self):
        children = self.get_family_relative('child')
        for child in children:
            self.model.run_conversation(self, child, num_days=1)

    def update_mem_decay(self, thresh=0.01, step_decay=0.05):
        """
            Method to model memory loss because of old age
            Args:
                * thresh: float. Minimum value memory stability S can have
                * step_decay: float. Value of S decay per each step
            Output:
                * Updated memory stability array for each language agent speaks
        """
        for lang in self.lang_stats:
            mem_stab = self.lang_stats[lang]['S']
            self.lang_stats[lang]['S'] = np.where(mem_stab > thresh,
                                                  np.maximum(mem_stab - step_decay, thresh),
                                                  thresh)

    def stage_1(self, ix_agent, num_days=10):
        super().stage_1(ix_agent, num_days=num_days)

    def stage_2(self, ix_agent):
        if self.model.nws.friendship_network[self]:
            self.speak_to_random_friend(ix_agent, num_days=7)

    def stage_3(self, ix_agent):
        # Pensioners, if alive, will lead family gathering
        if self.model.nws.family_network[self]:
            self.gather_family()
        self.stage_2(ix_agent)

    def stage_4(self, ix_agent, num_days=10):
        self.go_back_home()
        self.stage_1(ix_agent, num_days=num_days)
        self.stage_2(ix_agent)
        # speak to children ( action led by parents )
        self.speak_to_children()


