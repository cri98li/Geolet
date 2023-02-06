import numpy as np
from sklearn.exceptions import DataDimensionalityWarning
from tqdm.auto import tqdm

from Geolet.normalizers.NormalizerInterface import NormalizerInterface

class FirstPoint(NormalizerInterface):
    def _checkFormat(self, X):
        if X.shape[1] != 3:
            raise DataDimensionalityWarning(
                "The input data must be in this form [[part, latitude, longitude]]")

    def __init__(self, verbose=True):
        self.verbose = verbose

    def fit(self, X):
        self._checkFormat(X)

        return self

    """
    l'input sarà:
    latitude, longitude
    """

    def transform(self, part: np.ndarray, X: np.ndarray):
        X_res = X.copy()

        first_part = None
        first_coord = (None, None)
        for i in tqdm(range(len(part)), disable=not self.verbose, position=0, leave=True):
            if part[i] != first_part:
                first_part = part[i]
                first_coord = X[i]

            X_res[i] -= first_coord

        return X_res

    def _transformSingleTraj(self, X):
        X = X.copy()

        return X - X[0]

        #return list(map(lambda x: x - row[0], row))
