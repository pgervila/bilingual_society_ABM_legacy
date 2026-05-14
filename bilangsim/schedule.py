import random

from .agent import IndepAgent, Young


class StagedActivationModif:
    """Standalone staged-activation scheduler.

    Formerly subclassed mesa.time.StagedActivation, but bilangsim fully overrides
    step(); only the trivial bookkeeping from the Mesa base class is reproduced here.
    """

    def __init__(self, model, stage_list, shuffle=False, shuffle_between_stages=False):
        self.model = model
        self.steps = 0
        self.time = 0
        self.agents = []
        self.stage_list = stage_list
        self.shuffle = shuffle
        self.shuffle_between_stages = shuffle_between_stages
        self.stage_time = 1 / len(stage_list)

    def add(self, agent):
        self.agents.append(agent)

    def remove(self, agent):
        while agent in self.agents:
            self.agents.remove(agent)

    def get_agent_count(self):
        return len(self.agents)

    def step(self, pct_threshold=0.9):
        """ Executes all the stages for all agents """

        # TODO: following block should be a model method
        for ag in self.agents:
            # set conversation counters to zero when step begins
            ag._conv_counts_per_step = 0
            # new step -> older age
            ag.grow()
            for lang in self.model.langs:
                # save copy of wc for each agent
                ag.wc_init[lang] = ag.lang_stats[lang]['wc'].copy()
                ag.update_word_activation_elapsed_time(lang)
                ag.update_memory_retention(lang, pct_threshold=pct_threshold)
                ag.reset_step_mask(lang)
            # Update lang switch
            ag.update_lang_switch()
        if self.shuffle:
            random.shuffle(self.agents)

        # Network adj matrices will be constant through all stages of a given step
        self.model.nws.compute_adj_matrices()

        for stage in self.stage_list:
            for ix_ag, ag in enumerate(self.agents):
                if isinstance(ag, IndepAgent):
                    getattr(ag, stage)(ix_ag)
                else:
                    getattr(ag, stage)()
            self.time += self.stage_time

        # check reproduction, death : Need to make shallow copy of agents list,
        # since we are potentially removing agents as we iterate
        for ag in self.agents[:]:
            # once all speaking interactions are over in current step, copy final wc values
            for lang in self.model.langs:
                ag.wc_final[lang] = ag.lang_stats[lang]['wc'].copy()
            # An agent might have changed type if hired as Teacher to replace dead one
            # Old agent instance will be removed from model but it might still be in the copied list !
            try:
                if isinstance(ag, Young):
                    ag.reproduce()
                ag.random_death()
            except KeyError:
                pass

        if not self.steps % (2 * self.model.steps_per_year):
            self.model.update_friendships()

        # loop and update courses in schools and universities year after year
        # update jobs lang policy
        if self.steps and not self.steps % self.model.steps_per_year:
            self.model.update_centers()
            # Add immigration families if requested
            if self.model.pct_immigration:
                self.model.add_immigration()

        self.steps += 1

    def replace_agent(self, old_agent, new_agent):
        ix_in_schedule = self.model.schedule.agents.index(old_agent)
        self.model.schedule.remove(old_agent)
        self.model.schedule.agents.insert(ix_in_schedule, new_agent)

