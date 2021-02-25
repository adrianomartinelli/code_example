import networkx as nx
import pandas as pd
from utils.tools.graph import df_to_node_attr

class BaseGraphBuilder:

    def __init__(self, config: dict):
        '''\
        Contact-Graph Builder constructor

        Parameters
        ----------
        config: dict
            Dictionary containing `builder_params` and optionally `cellmask_file` if graph should be constructed from cellmask file alone.
        '''
        self.config = config

    def __call__(self, ndata: pd.DataFrame, edata: pd.DataFrame = None):
        self.ndata = ndata
        self.edata = edata
        self.graph = nx.Graph()

        self._add_nodes()
        self._add_nodes_attr()

        if edata is None:
            self._build_topology()
        else:
            self._add_edges()
            self._add_edges_attr()

        return self.graph

    def _add_nodes(self):
        self.graph.add_nodes_from(self.ndata.index)

    def _add_nodes_attr(self):
        attr = df_to_node_attr(self.ndata)
        nx.set_node_attributes(self.graph, attr)

    def _add_edges(self):
        self.graph.add_edges_from(self.edata.index)

    def _add_edges_attr(self):
        attr = df_to_node_attr(self.edata)
        nx.set_edge_attributes(self.graph, attr)

    def _build_topology(self):
        raise NotImplementedError('Implemented in subclasses.')

    # Convenient method to build graph from cellmask
    @classmethod
    def from_cellmask(cls, config: dict):

        # load required dependencies
        try:
            import numpy as np
            from skimage.io import imread
            # from skimage.measure import regionprops
            from skimage.measure import regionprops_table
        except ImportError:
            raise ImportError(
                'Please install the skimage: `conda install -c anaconda scikit-image`.')

        instance = cls(config)
        instance.cellmask = imread(config['cellmask_file'])

        # get object ids, 0 is background.
        objs = np.unique(instance.cellmask)
        objs = objs[objs != 0]

        # construct ndata
        # regions = regionprops(instance.cellmask)
        # pos = [region.centroid for region in regions]
        # ndata = pd.DataFrame({'pos': pos}, index=objs)

        ndata = regionprops_table(instance.cellmask, properties=['centroid'])
        ndata = pd.DataFrame(ndata, index=objs)
        ndata.columns = ['y', 'x'] # NOTE: axis 0 is y and axis 1 is x

        return instance(ndata)

