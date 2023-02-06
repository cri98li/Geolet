import numpy as np


class NormalizerInterface():

    def transform(self, part: np.ndarray, X: np.ndarray):
        pass
    def _transformSingleTraj(self, X):
        pass
