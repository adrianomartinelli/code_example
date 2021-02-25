from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.measure import regionprops
from skimage.morphology import binary_dilation

from graph_builder.base_graph_builder import BaseGraphBuilder
from graph_builder.constants import DILATION_KERNELS, EDGE_WEIGHT
# %%
class ContactGraphBuilder(BaseGraphBuilder):
    '''\
    Contact-Graph class.

    Build contact graph based on pixel expansion of cell masks.
    '''

    def __init__(self, config: dict):
        '''\
        Contact-Graph Builder constructor

        Parameters
        ----------
        config: dict
            Dictionary containing dict `builder_params` that specifies the dilation_kernel, radius and `cellmask_file`.
        '''
        super().__init__(config)

    def _build_topology(self):
        '''\
        Build topology using pixel expansion of cell masks. Cells which cell masks overlap after expansion are connected in the graph.
        '''

        # type hints
        self.graph: nx.Graph

        params = self.config['builder_params']

        if self.cellmask is None:
            self.cellmask = imread(self.config['cellmask_file'])

        if params['dilation_kernel'] in DILATION_KERNELS:
            kernel = DILATION_KERNELS[params['dilation_kernel']](params['radius'])
        else:
            raise ValueError(
                f'Specified dilate kernel not available. Please use one of {{{", ".join(DILATION_KERNELS)}}}.')

        # get object ids, 0 is background.
        objs = np.unique(self.cellmask)
        objs = objs[objs != 0]

        # compute neighbours
        edges = []
        for obj in objs:
            dilated_img = binary_dilation(self.cellmask == obj, kernel)
            cells = np.unique(self.cellmask[dilated_img])
            cells = cells[cells != obj]  # remove object itself
            cells = cells[cells != 0]  # remove background
            edges.extend([(obj, cell, {EDGE_WEIGHT: 1}) for cell in cells])

        self.graph.add_edges_from(edges)