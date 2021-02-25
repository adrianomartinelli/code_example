from skimage.morphology import square, diamond, disk

DILATION_KERNELS = {
    'disk': disk,
    'square': square,
    'diamon': diamond
}

EDGE_WEIGHT = 'weight'