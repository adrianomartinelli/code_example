from scipy.spatial.distance import pdist
from collections.abc import Iterable
import numpy as np

class RipleysK:
    '''
    Estimates Ripley's K for TMA samples

    Parameters
    ----------
    height: scalar
        height of the TMA, used to compute area.
    width: scalar
        width of the TMA, used to compute area.
    X : 2D array
        An n x 2 array with coordinates of cells.
    T : 1D array, scalar
        Distance(s) at which Ripley's K is evaluated. Should be T < (area/2)**0.5
    correction : str, None
        Correction to compensate for edge effects

    Returns
    -------
    res : 1D array
        Ripley's K estimator evaluated at T.

    References
    ----------
    .. [1] http://doi.wiley.com/10.1002/9781118445112.stat07751
    '''

    VALID_CORRECTIONS = [None, 'ripley']

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.area = height * width

    def fit(self, X, metric='euclidean'):
        self.N = len(X)  # number of observations
        self.dists = pdist(X, metric=metric)  # all pair-wise distances

    def predict(self, T, correction=None):
        T = self._process_check_T(T)

        if correction not in self.VALID_CORRECTIONS:
            raise ValueError(f'Invalid correction method `{correction}`. Available methods are: {self.VALID_CORRECTIONS}')
        if correction:
            # TODO: implement Ripleys correction method
            raise NotImplementedError('To be implemented.')

        res = np.zeros(len(T))
        for i, t in enumerate(T):
            res[i] = (self.dists < t).sum()

        return 2 * self.area * res / self.N ** 2

    def L_estimator(self, res):
        return np.sqrt(res / np.pi)

    def csr_test(self, T, correction=None):
        res = self.predict(T, correction)
        L = self.L_estimator(res)
        return L - T

    def _get_dist(self, i, j):
        if i == j:
            return 0
        elif j < i:
            tmp = j
            j = i
            i = tmp
        return self.dists[self.N * i + j - ((i + 2) * (i + 1)) // 2]

    def _process_check_T(self, T):
        T = self._make_iterable(T)
        T = np.fromiter(T, float)
        if np.any(T < 0):
            raise ValueError('T cannot contain negative values.')
        return T

    def csr(self, T):
        T = self._process_check_T(T)
        return np.pi * T ** 2

    @staticmethod
    def _make_iterable(obj):
        '''

        Parameters
        ----------
        obj : Any
            Any object that you want to make iterable

        Returns
        -------
        Packed object, possible to iterate overs
        '''

        # if object is iterable and its not a string return object as is
        if isinstance(obj, Iterable) and not isinstance(obj, str):
            return obj
        else:
            return (obj,)


# %% example with random data
import pandas as pd
import matplotlib.pyplot as plt

height = 10
width = 10
area = height * width
df = pd.DataFrame({'x': np.random.random((1000,)) * width,
                   'y': np.random.random((1000,)) * height,
                   'meta_id': np.repeat(1, 1000)})

X = df[['x', 'y']]
T = np.linspace(0, np.sqrt(area / 2))
rke = RipleysK(height, width)
rke.fit(X)
res = rke.predict(T)
csr = rke.csr(T)
csr_test_res = rke.csr_test(T)

# %%
n = range(len(T))
fig, axes = plt.subplots(1, 2, dpi=300)
axes[0].plot(n, res, marker='o', linestyle='-', markersize=3, label=r'$\hat{K}(t)$')
axes[0].plot(n, csr, linestyle='--', label=r'$\hat{K}_{CSR}(t)$')
axes[0].set_title("Ripley's K")

axes[1].plot(n, csr_test_res, marker='o', markersize=3, linestyle='-', label=r'$\hat{L}(t) - t$')
axes[1].plot(n, [0]*len(T), linestyle='--', label=r'$L(t) - t$')
axes[1].set_title("Test for deviation from\nhomogeneous Poisson process")

for ax in axes:
    ax.set_xlabel('T')
    ax.legend()

fig.tight_layout()
fig.savefig('ripleysK.png')
fig.show()
