import random
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from sklearn.preprocessing import normalize
import warnings
from sklearn.utils import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

from .agent import Child, Adolescent, Young, Adult, Teacher


class NetworkBuilder:
    """ Class to deal with everything that has to do with networks in model """

    family_mirror = {'consort': 'consort', 'father': 'child', 'mother': 'child',
                     'grandfather': 'grandchild', 'grandmother': 'grandchild',
                     'uncle': 'nephew', 'aunt': 'nephew', 'sibling': 'sibling',
                     'cousin': 'cousin'}

    def __init__(self, model):

        self.adj_mat_fam_nw = None
        self.adj_mat_friend_nw = None
        self.model = model
        # setup
        self.create_networks()
        self.add_ags_to_networks(self.model.schedule.agents)

    def create_networks(self):
        # INITIALIZE KNOWN PEOPLE NETWORK => label is lang spoken
        self.known_people_network = nx.DiGraph()
        # INITIALIZE FRIENDSHIP NETWORK
        self.friendship_network = nx.Graph()
        # INITIALIZE FAMILY NETWORK
        self.family_network = nx.DiGraph()
        # INITIALIZE JOBS NETWORK
        self.jobs_network = nx.Graph()

    def add_ags_to_networks(self, ags, *more_ags):
        """
            Method that adds agents to all networks in model
            Args:
               * ags: single agent instance or list of agents instances
        """
        ags = [ags] if type(ags) != list else ags
        for nw in [self.known_people_network,
                   self.friendship_network,
                   self.family_network]:
            nw.add_nodes_from(ags)

    def build_networks(self):
        self.define_family_networks()
        self.define_friendship_networks()
        self.define_jobs_network()

    def define_family_links(self, family):
        """ Method to define all links between all members of a given family
            Args:
                * family: list of agent instances
        """
        # find out lang of interaction btw family members
        (lang_consorts, lang_with_father,
         lang_with_mother, lang_siblings) = self.model.get_lang_fam_members(family)
        # initialize family network
        # add family edges in family and known_people networks ( both are DIRECTED networks ! )
        self.set_link_with_relatives(family[0], family[1],
                                     'consort', lang_with_relatives=lang_consorts)
        self.set_link_with_relatives([family[2], family[3]], family[0],
                                     'father', lang_with_relatives=lang_with_father)
        self.set_link_with_relatives([family[2], family[3]], family[1],
                                     'mother', lang_with_relatives=lang_with_mother)
        self.set_link_with_relatives(family[2], family[3],
                                     'sibling', lang_with_relatives=lang_siblings)

    def define_family_networks(self):
        """
            Method to define family links between agents. It also adds relatives to known_people_network
            It assumes that the distribution of languages in clusters has already been sorted at cluster level or
            by groups of four to ensure linguistic affinity within families.
            Marriage, to make things simple, only allowed for linguistic combinations ->  0-1, 1-1, 1-2
            Once all family links are defined, method also assigns school jobs to parents so that entire family
            will move if needed
        """
        for clust_idx, clust_info in self.model.geo.clusters_info.items():
            # trick to iterate over groups of agents of size = self.model.family_size
            for idx, family in enumerate(zip(*[iter(clust_info['agents'])] * self.model.family_size)):
                self.define_family_links(family)
            # set up agents left out of family partition of cluster
            len_clust = len(clust_info['agents'])
            num_left_agents = len_clust % self.model.family_size
            if num_left_agents:
                for idx2, ag in enumerate(clust_info['agents'][-num_left_agents:]):
                    ag.set_lang_ics()
        # assign school jobs
        self.model.geo.assign_school_jobs()

    def define_friendship_networks(self, init_max_num_friends=10):
        """ """
        # TODO : Apply small world graph to relevant nodes using networkx
        for ag in self.model.schedule.agents:
            num_friends = len(self.friendship_network[ag])
            max_num_friends = ag.info['max_num_friends']
            if type(ag) in (Adult, Young):
                if ag.loc_info['job']:
                    info_occupation = ag.loc_info['job'].info
                else:
                    clust = ag['clust']
                    info_occupation = random.choice(self.model.geo.clusters_info[clust]['jobs']).info
                colleagues = 'employees'
            elif isinstance(ag, Teacher):
                if ag.loc_info['job'][0]:
                    info_occupation = ag.loc_info['job'][0].info
                colleagues = 'employees'
            elif isinstance(ag, (Child, Adolescent)):
                school, course_key = ag.loc_info['school']
                info_occupation = school.grouped_studs[course_key]
                colleagues = 'students'
            else:
                continue
            for coll in info_occupation[colleagues].difference({ag}):
                if len(self.friendship_network[ag]) >= min(max_num_friends, init_max_num_friends):
                    break
                # check colleague lang distance and all friendship conditions
                if ag.check_friend_conds(coll, init_max_num_friends=init_max_num_friends):
                    ag.make_friend(coll)


    def define_jobs_network(self, p=0.3):
        """ Defines random graph with probability p where nodes are job instances"""
        jobs = [j for cl in range(self.model.geo.num_clusters)
                for j in self.model.geo.clusters_info[cl]['jobs']]
        self.jobs_network = nx.gnp_random_graph(len(jobs), p)
        nodes_mapping = {i: j for i, j in enumerate(jobs)}
        self.jobs_network = nx.relabel_nodes(self.jobs_network, nodes_mapping)

    def set_newborn_family_links(self, agent, father, mother, lang_with_father, lang_with_mother):
        """
            Method to define family links and interaction language of a newborn agent.
            Corresponding edges are created in family and known people networks
            Args:
                * agent: agent instance. The agent on which links have to be defined
                * father: agent instance. The agent's father
                * mother: agent instance. The agent's mother
                * lang_with_father: integer [0, 1]. Defines lang of agent with father
                * lang_with_mother: integer [0, 1]. Defines lang of agent with mother
        """
        fam_nw = self.family_network

        # get agent siblings from mother BEFORE agent birth
        siblings = mother.get_family_relative('child')
        # Define family links with parents
        self.set_link_with_relatives(agent, (father, mother), ('father', 'mother'),
                                     lang_with_relatives=(lang_with_father, lang_with_mother))
        # Define links with siblings if any
        self.set_link_with_relatives(agent, siblings, 'sibling')

        # rest of family will if possible speak same language with baby as their link agent to the baby
        for elder, lang in zip([father, mother], [lang_with_father, lang_with_mother]):
            for relat, labels in fam_nw[elder].items():
                lang_label = 'L1' if lang == 0 else 'L2'
                if relat.lang_stats[lang_label]['pct'][relat.info['age']] > relat.lang_thresholds['speak']:
                    com_lang = lang
                else:
                    com_lang = 1 if lang == 0 else 0
                if labels["fam_link"] == 'father':
                    self.set_link_with_relatives(agent, relat, 'grandfather',
                                                 lang_with_relatives=com_lang)
                elif labels["fam_link"] == 'mother':
                    self.set_link_with_relatives(agent, relat, 'grandmother',
                                                 lang_with_relatives=com_lang)
                elif labels["fam_link"] == 'sibling':
                    self.set_link_with_relatives(agent, relat,
                                                 'uncle' if relat.info['sex'] == 'M' else 'aunt',
                                                 lang_with_relatives=com_lang)
                    if isinstance(relat, Young) and relat.info['married']:
                        consort = relat.get_family_relative('consort')
                        self.set_link_with_relatives(agent, consort,
                                                     'uncle' if consort.info['sex'] == 'M' else 'aunt',
                                                     lang_with_relatives=com_lang)
                elif labels["fam_link"] == 'nephew':
                    self.set_link_with_relatives(agent, relat, 'cousin',
                                                 lang_with_relatives=com_lang)

    def set_link_with_relatives(self, agents, relatives, labels, lang_with_relatives=None):
        """ Method to set network links with relatives
            Args:
                * agents: agent instance or list/tuple of class instances. Agents on which
                    to set up family links
                * relatives: agent instance or list/tuple of class instances. Corresponding
                    relative(s) of agent(s).
                * labels: string or list/tuple of strings. Characterizes link between agents and relatives.
                    If all links are the same, a unique string can be specified
                * lang_with_relatives: tuple of integers. Optional. If None,
                    the language between agent and relative will be relatives' best known language.
                    Default None
            Output:
                *
        """
        fam_nw = self.family_network
        kn_people_nw = self.known_people_network

        if not isinstance(agents, (tuple, list)):
            agents = [agents]
        if not isinstance(relatives, (tuple, list)):
            relatives = [relatives]
        if isinstance(labels, str):
            labels = (labels, ) * len(relatives)
        if not lang_with_relatives:
            lang_with_relatives = [relative.get_dominant_lang() for relative in relatives]
        else:
            if not isinstance(lang_with_relatives, (tuple, list)):
                lang_with_relatives = (lang_with_relatives,) * len(relatives)
        for agent in agents:
            for relative, fam_link, lang in zip(relatives, labels, lang_with_relatives):
                for (i, j, link) in [(agent, relative, fam_link),
                                     (relative, agent, self.family_mirror[fam_link])]:
                    fam_nw.add_edge(i, j, fam_link=link, lang=lang)
                    kn_people_nw.add_edge(i, j, family=True, lang=lang)

    def compute_adj_matrices(self):
        """
            Method to compute adjacent matrices for family and friends
            Output:
                * Method sets value of adj_mat_fam_nw and adj_mat_friend_nw NetworkBuilder
                    class attributes
        """
        # modern networkx returns a csr_array, whose single-row indexing yields a
        # coo_array (no .indices/.data); wrap in the legacy csr_matrix so that
        # downstream row-slicing keeps CSR semantics
        fam_graph = nx.adjacency_matrix(self.model.nws.family_network,
                                        nodelist=self.model.schedule.agents)
        self.model.nws.adj_mat_fam_nw = csr_matrix(normalize(fam_graph, norm='l1', axis=1))
        # compute adjacent matrices for friends
        friend_graph = nx.adjacency_matrix(self.model.nws.friendship_network,
                                           nodelist=self.model.schedule.agents)
        self.model.nws.adj_mat_friend_nw = csr_matrix(normalize(friend_graph, norm='l1', axis=1))

    def plot_family_networks(self):
        """ PLOT NETWORK with lang colors and position """
        #        people_pos = [elem.pos for elem in self.schedule.agents if elem.agent_type == 'language']
        #        # Following works because lang agents are added before other agent types
        #        people_pos_dict= dict(zip(self.schedule.agents, people_pos))
        people_pos_dict = {elem: elem.pos
                           for elem in self.model.schedule.agents}
        people_color = [elem.language for elem in self.family_network]
        nx.draw(self.family_network, pos=people_pos_dict, node_color=people_color)

    def plot_jobs_network(self):
        jobs_pos = {j: j.pos for cl in range(self.model.geo.num_clusters)
                    for j in self.model.geo.clusters_info[cl]['jobs']}
        nx.draw(self.jobs_network, pos=jobs_pos)

    def __getstate__(self):
        state = self.__dict__.copy()
        for nw in ['known_people_network', 'family_network', 'friendship_network', 'jobs_network']:
            state[nw] = [tuple(state[nw].nodes()), tuple(state[nw].edges(data=True))]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.known_people_network = nx.DiGraph()
        self.known_people_network.add_nodes_from(state['known_people_network'][0])
        self.known_people_network.add_edges_from(state['known_people_network'][1])
        self.family_network = nx.DiGraph()
        self.family_network.add_nodes_from(state['family_network'][0])
        self.family_network.add_edges_from(state['family_network'][1])
        self.friendship_network = nx.Graph()
        self.friendship_network.add_nodes_from(state['friendship_network'][0])
        self.friendship_network.add_edges_from(state['friendship_network'][1])
        self.jobs_network = nx.Graph()
        self.jobs_network.add_nodes_from(state['jobs_network'][0])
        self.jobs_network.add_edges_from(state['jobs_network'][1])



