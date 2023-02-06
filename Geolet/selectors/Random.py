import random

import numpy as np

from Geolet.normalizers.NormalizerInterface import NormalizerInterface
from Geolet.selectors.SelectorInterface import SelectorInterface


class Random(SelectorInterface):

    def __init__(self, normalizer: NormalizerInterface, n_geolet_per_class=10, verbose=True):
        self.verbose = verbose
        self.n_geolets = n_geolet_per_class
        self.normalizer = normalizer

    def fit(self, X):
        return self

    """
    l'output sarà:
    tid, class, time, c1, c2, geohash
    """

    def transform(self, tid: np.ndarray, classes: np.ndarray, time: np.ndarray, X: np.ndarray, partid: np.ndarray):
        selected = []
        pk_array = np.array([(a,b) for a, b in zip(tid, partid)])

        for classe in np.unique(classes):
            to_choice = np.unique(pk_array[classes == classe], axis=0).tolist()

            n = min(self.n_geolets, len(to_choice))

            choices = random.sample(to_choice, n)
            selected.append(choices)

        selected = [el for lista in selected for el in lista]

        to_keep_indeces = (pk_array[:, None] == selected).all(-1).any(-1) #TODO BUG?

        X[to_keep_indeces] = self.normalizer.transform(pk_array[to_keep_indeces].tolist(), X[to_keep_indeces])
        time[to_keep_indeces] = self.normalizer.transform(pk_array[to_keep_indeces].tolist(), time[to_keep_indeces])

        pk_array = np.array([f"{a}{b}" for a, b in pk_array])

        return pk_array[to_keep_indeces], classes[to_keep_indeces], time[to_keep_indeces], X[to_keep_indeces]