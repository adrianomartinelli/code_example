from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np

from graph_builder.base_graph_builder import BaseGraphBuilder
from graph_builder.constants import EDGE_WEIGHT
# %%
class KNNGraphBuilder(BaseGraphBuilder):
    '''\
    KNN (K-Nearest Neighbors) class for graph building.
    '''

    def __init__(self, config: dict):
        '''\
        KNN-Graph Builder constructor

        Parameters
        ----------
        config: dict
            Dictionary containing `builder_params`.
            Refere to [1] for possible parameters

        Notes
        _____
        Example `builder_params` :
            config = {'builder_params': {'n_neighbors': 5, 'mode':'connectivity', 'metric':'minkowski', 'p':2, 'n_jobs':-1}, 'cellmask_file': 'path_to_file'}

        See Also
        ________
        .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
        '''
        super().__init__(config)

    def _build_topology(self):
        '''\
        Build topology using a kNN algorithm based on the distance between the centroid of the nodes.
        '''

        # type hints
        self.graph: nx.Graph

        # compute adjacency matrix
        adj = kneighbors_graph(self.ndata, **self.config['builder_params'])

        # construct and add edges
        # IMPORTANT: We need to map the U,V indices to the object_ids!
        node_map = {idx: self.ndata.index[idx] for idx in range(len(self.ndata))}

        U, V = np.nonzero(adj)
        edges = [(node_map[u],node_map[v], {EDGE_WEIGHT:adj[u,v]}) for u,v in zip(U,V)]

        self.graph.add_edges_from(edges)