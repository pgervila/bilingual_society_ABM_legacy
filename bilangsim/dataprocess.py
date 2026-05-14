import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import dill

from .agent import Baby, Child, Adolescent, Young, YoungUniv
from .agent import Adult, Teacher, TeacherUniv, Pensioner


class DataProcessor:
    def __init__(self, model, save_dir=''):
        self.model = model
        self.model_data = None
        self.model_reporters = {"pct_spa": lambda dp: dp.get_lang_stats(0),
                                "pct_bil": lambda dp: dp.get_lang_stats(1),
                                "pct_cat": lambda dp: dp.get_lang_stats(2),
                                "total_num_agents": lambda dp: len(dp.model.schedule.agents),
                                "pct_cat_in_biling": lambda dp: dp.get_global_bilang_inner_evol()
                                }
        self.agent_reporters = {"pct_cat_knowledge": lambda a: a.lang_stats['L2']['pct'][a.info['age']],
                                "pct_L21_knowledge": lambda a: a.lang_stats['L21']['R'].mean(),
                                "pct_spa_knowledge": lambda a: a.lang_stats['L1']['pct'][a.info['age']],
                                "pct_L12_knowledge": lambda a: a.lang_stats['L12']['R'].mean(),
                                "tokens_per_step_spa": lambda a: (a.wc_final['L1'] - a.wc_init['L1']).sum(),
                                "tokens_per_step_cat": lambda a: (a.wc_final['L2'] - a.wc_init['L2']).sum(),
                                "x": lambda a: a.pos[0],
                                "y": lambda a: a.pos[1],
                                "age": lambda a: a.info['age'],
                                "language": lambda a: a.info['language'],
                                "excl_c": lambda a: a.lang_stats['L1']['excl_c'][a.info['age']] if a.info['language'] == 2 else a.lang_stats['L2']['excl_c'][a.info['age']],
                                "clust_id": lambda a: a.loc_info['home'].info['clust'],
                                "agent_type": lambda a: type(a).__name__,
                                "num_conv_step": lambda a: a._conv_counts_per_step
                                }
        self.model_vars = {k: [] for k in self.model_reporters}
        self.agent_vars = {k: [] for k in self.agent_reporters}

        self.init_conds = {'num_clusters': self.model.geo.num_clusters,
                           'cluster_sizes': self.model.geo.cluster_sizes,
                           'cluster_centers': self.model.geo.clust_centers,
                           'init_num_people': self.model.num_people,
                           'grid_width': self.model.grid.width,
                           'grid_height': self.model.grid.height,
                           'init_lang_distrib': self.model.init_lang_distrib,
                           'sort_by_dist': self.model.lang_ags_sorted_by_dist,
                           'sort_within_clust': self.model.lang_ags_sorted_in_clust}

        self.save_dir = save_dir

    def collect(self):
        """ Collect all the data for the given model object. """
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(self))

        if self.agent_reporters:
            for var, reporter in self.agent_reporters.items():
                agent_records = []
                for agent in self.model.schedule.agents:
                    agent_records.append((agent.unique_id, reporter(agent)))
                self.agent_vars[var].append(agent_records)

    def get_model_vars_dataframe(self):
        """ Create a pandas DataFrame from the model variables.

        The DataFrame has one column for each model variable, and the index is
        (implicitly) the model tick.

        """
        curr_step = self.model.schedule.steps
        len_stored_data = len(self.model_vars['pct_bil'])
        return pd.DataFrame(self.model_vars,
                            index=range(curr_step-len_stored_data, curr_step))

    def get_agent_vars_dataframe(self):
        """ Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        """
        data = defaultdict(dict)
        for var, records in self.agent_vars.items():
            curr_step = self.model.schedule.steps
            len_stored_data = len(self.agent_vars['age'])
            for step, entries in zip(range(curr_step-len_stored_data, curr_step), records):
                for entry in entries:
                    agent_id = entry[0]
                    val = entry[1]
                    data[(step, agent_id)][var] = val
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.names = ["Step", "AgentID"]
        return df

    def get_agent_by_id(self, ag_id):
        return [ag for ag in self.model.schedule.agents if ag.unique_id == ag_id][0]

    def get_agents_by_type(self, agent_type):
        return [ag for ag in self.model.schedule.agents if type(ag) == agent_type]

    def get_tokens_per_step(self, ag, step):
        tot_tokens_per_step = dict()
        for stage, vals in ag._words_per_conv_counts[step].items():
            tot_tokens_per_stage = 0
            for tups in vals:
                tot_words = self.model.num_words_conv[tups[0]] * tups[1]
                tot_tokens_per_stage += tot_words
            tot_tokens_per_step[stage] = tot_tokens_per_stage
        return tot_tokens_per_step

    @staticmethod
    def get_conv_type_counter(ag, step, stage):
        return Counter(ag._words_per_conv_counts[step][stage])

    def get_tokens_stats_per_type(self, ag_type, step):
        agents = self.get_agents_by_type(ag_type)
        df_tokens_per_stage = pd.DataFrame.from_dict([self.get_tokens_per_step(ag, step)
                                                      for ag in agents])
        return df_tokens_per_stage.describe()

    def get_lang_stats(self, lang_type):
        """Method to get counts of each type of lang agent

        Arguments:
            * lang_type : integer from [0,1,2] that specifies agent linguistic type

        Returns:
            * lang type count as percentage of total

        """
        ag_lang_list = [ag.info['language'] for ag in self.model.schedule.agents]
        lang_counts = Counter(ag_lang_list)
        return lang_counts[lang_type] / len(ag_lang_list)

    def get_global_bilang_inner_evol(self):
        """ Method to compute internal linguistic structure of all bilinguals,
        expressed as average amount of Catalan heard or spoken as % of total

         Output:
             * float representing the AVERAGE percentage of Catalan in bilinguals

        """
        list_biling = [ag.lang_stats['L2']['pct'][ag.info['age']]
                       for ag in self.model.schedule.agents
                       if ag.info['language'] == 1]
        if list_biling:
            return np.array(list_biling).mean()
        else:
            if self.get_lang_stats(2) > self.get_lang_stats(0):
                return 1
            else:
                return 0

    def get_tokens_per_day(self):
        pass

    def get_agents_attrs_value(self, ag_attr, plot=False):
        """ Get value of specific attribute for all lang agents in model
            Args:
                * ag_attr: string. Specifies numerical agent attribute
                * plot: boolean. True to plot processed info, False otherwise
            Output:
                * Assigns processed value to 'df_attrs_avg'
        """
        ag_and_coords = [(ag.info[ag_attr], ag.pos[0], ag.pos[1])
                         for ag in self.model.schedule.agents]
        ag_and_coords = np.array(ag_and_coords)
        df_attrs = pd.DataFrame({'values': ag_and_coords[:, 0],
                                 'x': ag_and_coords[:, 1],
                                 'y': ag_and_coords[:, 2]})
        self.df_attrs_avg = df_attrs.groupby(['x', 'y']).mean()

        if plot:
            s = plt.scatter(self.df_attrs_avg.reset_index()['x'],
                            self.df_attrs_avg.reset_index()['y'],
                            c=self.df_attrs_avg.reset_index()['values'],
                            marker='s',
                            vmin=0, vmax=2, s=30,
                            cmap='viridis')
            plt.colorbar(s)
            plt.show()

    def get_attrs_avg_map(self):
        """
            Method to store averaged attribute values per cell
            into a numpy array
        """
        attr_map = np.full((self.model.grid_width, self.model.grid_height), np.nan)
        for idx, val in self.df_attrs_avg.iterrows():
            attr_map[idx[::-1]] = val['values']
        plt.imshow(attr_map, vmin=0, vmax=2, cmap='viridis', interpolation="nearest")
        plt.colorbar()

    def pickle_model(self):
        with open('pickled_model', 'wb') as f:
            dill.dump(self.model, f, byref=True)

    @staticmethod
    def unpickle_model(filename):
        """ Method to unpickle a pickled model
            Args:
                * filename: string. Name of pickle file
        """

        with open(filename, 'rb') as f:
            unpickled_model = dill.load(f)

        # Convert lists into sets so that model can be computed
        for k, cl in unpickled_model.geo.clusters_info.items():
            for h in cl['homes']:
                h.agents_in = set(h.agents_in)
                h.info['occupants'] = set(h.info['occupants'])
            for sc in cl['schools']:
                sc.agents_in = set(sc.agents_in)
                sc.info['employees'] = set(sc.info['employees'])
                sc.info['students'] = set(sc.info['students'])
                for course in sc.grouped_studs.values():
                    course['students'] = set(course['students'])
                sc.agents_in = set(sc.agents_in)
            for job in cl['jobs']:
                job.agents_in = set(job.agents_in)
                job.info['employees'] = set(job.info['employees'])
            if 'university' in cl:
                univ = cl['university']
                univ.info['employees'] = set(univ.info['employees'])
                univ.info['students'] = set(univ.info['students'])
                for fac in univ.faculties.values():
                    fac.agents_in = set(fac.agents_in)
                    fac.info['employees'] = set(fac.info['employees'])
                    fac.info['students'] = set(fac.info['students'])
                    for course in fac.grouped_studs.values():
                        course['students'] = set(course['students'])

        return unpickled_model

    def save_model_data(self, save_data_freq):
        # get dataframes from stored data
        df_model_data = self.get_model_vars_dataframe()
        df_agent_data = self.get_agent_vars_dataframe()
        # save dataframes
        save_path = os.path.join(self.save_dir, 'model_data.h5')
        # TODO: add reinitialization option
        if self.model.schedule.steps == save_data_freq:
            with pd.HDFStore(save_path, mode='w') as hdf_db:
                hdf_db.append('model_data', df_model_data, format='t', data_columns=True)
                hdf_db.append('agent_data', df_agent_data, format='t', data_columns=True,
                              min_itemsize={'agent_type': 11})
                ics = pd.DataFrame.from_dict(self.init_conds, orient='index').T
                ics.to_hdf(save_path, key='init_conds')
        else:
            df_model_data.to_hdf(save_path, key='model_data',
                                 append=True, mode='r+', format='table', data_columns=True)
            df_agent_data.to_hdf(save_path, key='agent_data',
                                 append=True, mode='r+', format='table', data_columns=True)
        # empty data to avoid keeping already-saved data in RAM
        self.model_vars = {k: [] for k in self.model_vars}
        self.agent_vars = {k: [] for k in self.agent_vars}

    @staticmethod
    def load_model_data(data_filename, key='/'):
        return dd.io.load(data_filename, key)

    def optimize_data_saving_space(self, filename='model_data.h5'):
        save_path = os.path.join(self.save_dir, filename)
        opt_save_path = os.path.join(self.save_dir, 'opt_model_data.h5')
        with pd.HDFStore(save_path) as f:
            for n in f.keys():
                data = pd.read_hdf(f, n)
                data.to_hdf(opt_save_path, n)
        os.remove(save_path)
        os.rename(opt_save_path, save_path)


class DataViz:
    """ Class with methods to visualize results"""
    def __init__(self, model):
        self.model = model

    def show_results(self, ag_attr='language', step=None,
                     plot_results=True, plot_type='scatter', save_fig=False):
        """ Method to ..."""
        grid_size = (3, 5)
        self.model.data_process.get_agents_attrs_value(ag_attr)

        data_2_plot = self.model.data_process.get_model_vars_dataframe()[:step]
        data_2D = self.model.data_process.df_attrs_avg.reset_index()

        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        data_2_plot[["pct_bil", "pct_cat", "pct_spa"]].plot(ax=ax1,
                                                            title='lang_groups',
                                                            color=['darkgreen','y','darkblue'])
        ax1.xaxis.tick_bottom()
        ax1.tick_params('x', labelsize='small')
        ax1.tick_params('y', labelsize='small')
        ax1.legend(loc='best', prop={'size': 8})
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        data_2_plot['total_num_agents'].plot(ax=ax2, title='num_agents')
        ax2.tick_params('x', labelsize='small')
        ax2.tick_params('y', labelsize='small')
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        data_2_plot['pct_cat_in_biling'].plot(ax=ax3, title='biling_quality')
        ax3.tick_params('x', labelsize='small')
        ax3.tick_params('y', labelsize='small')
        ax3.legend(loc='best', prop={'size': 8})
        ax4 = plt.subplot2grid(grid_size, (0, 0), rowspan=3, colspan=3)
        if plot_type == 'imshow':
            data = self.model.data_process.df_attrs_avg.unstack('x')
            s = ax4.imshow(data, vmin=0, vmax=2, cmap='viridis',
                           interpolation='nearest', origin='lower')
            ax4.set_xlim(0, data.shape[0])
            ax4.set_ylim(0, data.shape[1])

        else:
            s = ax4.scatter(data_2D['x'], data_2D['y'],
                            c=data_2D['values'], marker='s',
                            vmin=0, vmax=2, s=25, cmap='viridis')
            ax4.set_xlim(0, 100)
            ax4.set_ylim(0, 100)
        ax4.text(0.02, 1.04, 'time = %.1f' % self.model.schedule.steps, transform=ax4.transAxes)

        plt.colorbar(s)
        plt.suptitle(self.model.data_process.save_dir)
        plt.tight_layout()
        if save_fig:
            if self.model.data_process.save_dir:
                fig_path = os.path.join(self.model.data_process.save_dir,
                                        'step{}.png'.format(str(step)))
                plt.savefig(fig_path)
                plt.close()
            else:
                plt.savefig('step{}.png'.format(str(step)))
                plt.close()
        if plot_results:
            plt.show()


class VizImpData:

    def __init__(self, file_name=None, key='/'):
        self.data = dd.io.load(file_name, key)

    def show_imported_results(self, key='/'):
        pass


class PostProcessor:
    def __init__(self, data_filename, save_dir=''):
        filepath = os.path.join(save_dir, data_filename)
        with pd.HDFStore(filepath) as self.data:
            self.agent_data = self.data['agent_data']
            self.model_data = self.data['model_data']
            self.init_conditions = self.data['init_conds']

    def ag_results_by_id(self, ag_id):
        """ Args:
                * ag_id: integer. Agent unique id
        """
        idx = pd.IndexSlice
        return self.agent_data.unstack().loc[idx[:], idx[:, ag_id]]

    def ag_results_by_type(self, ag_type):
        """ Args:
                * ag_type: string. Agent class type
        """
        return self.agent_data[self.agent_data['agent_type'] == ag_type].unstack()
    # self.agent_data[self.agent_data['agent_type'] == 'Child'][filter_tokens].sum(1).unstack()
    # self.agent_data[self.agent_data['agent_type'] == 'Child'][filter_tokens].sum(1).unstack().mean().mean()

    def get_grouped_lang_stats(self, step, group_by='agent_type'):
        return self.agent_data.loc[step, :].groupby(group_by)[['pct_cat_knowledge', 'pct_spa_knowledge',
                                                               'pct_L21_knowledge', 'pct_L12_knowledge']].describe()

    def plot_lang_trend(self):
        return self.model_data[['pct_bil', 'pct_spa', 'pct_cat']].plot(grid=True)

    def get_num_tokens_per_agent_type(self, step):
        idx = pd.IndexSlice
        df_num_tokens = self.agent_data[
            ['tokens_per_step_spa', 'tokens_per_step_cat', 'agent_type']
                                       ].loc[idx[step, :], idx[:]]
        df_num_tokens['total_tokens'] = df_num_tokens[['tokens_per_step_spa',
                                                       'tokens_per_step_cat']].sum(1)
        return df_num_tokens.groupby('agent_type').total_tokens.agg(['max', 'min', 'mean'])

    def group_by_step_and_agent_type(self, *cols, method='mean'):
        grouped_data = self.agent_data[[*cols,
                                        'agent_type']].groupby(['Step', 'agent_type'])
        aggregated_data = getattr(grouped_data, method)()
        return aggregated_data.unstack()

    def group_by_step_and_language_type(self, *cols, method='mean'):
        grouped_data = self.agent_data[[*cols,
                                        'language']].groupby(['Step', 'language'])
        aggregated_data = getattr(grouped_data, method)()
        return aggregated_data.unstack()

    def plot_population_size(self, max_num_steps):
        self.agent_data.count(level='Step').iloc[:max_num_steps].age.plot(grid=True)
