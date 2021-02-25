import graph_builder as gb

# build cell-cell graph from cell_mask file
config = {'builder_params': {'dilation_kernel': 'disk', 'radius': 4},
          'cellmask_file': '/Users/art/Documents/thesis/data/OMEnMasks/Basel_Zuri_masks/BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_90_88_X11Y5_242_a0_full_maks.tiff'}
cellGraph = gb.ContactGraphBuilder.from_cellmask(config)

# build kNN graph from cell_mask file
config = {'builder_params': {'n_neighbors': 5, 'mode':'connectivity', 'metric':'minkowski', 'p':2, 'metric_params':None, 'include_self':False, 'n_jobs':-1},
          'cellmask_file': '/Users/art/Documents/thesis/data/OMEnMasks/Basel_Zuri_masks/BaselTMA_SP41_15.475kx12.665ky_10000x8500_5_20170905_90_88_X11Y5_242_a0_full_maks.tiff'}
kNNGraph = gb.KNNGraphBuilder.from_cellmask(config)
