from graph_builder.contact_graph_builder import ContactGraphBuilder
from graph_builder.knn_graph_builder import KNNGraphBuilder

GRAPH_BUILDERS = {
    'knn': KNNGraphBuilder,
    'contact': ContactGraphBuilder
}

GRAPH_BUILDER_DEFAULT_PARAMS = {
    'knn': {'builder_params': {'n_neighbors': 5, 'mode':'connectivity', 'metric':'minkowski', 'p':2, 'metric_params':None, 'include_self':False, 'n_jobs':-1}},
    'contact': {'builder_params': {'dilation_kernel': 'disk', 'radius': 4}}
}