from geolib import geohash
from sklearn.exceptions import *
from tqdm.auto import tqdm
import numpy as np

from Geolet.partitioners.PartitionerInterface import PartitionerInterface


class Geohash(PartitionerInterface):
    """
    precision : number, default=7
    A value used by the partitioner to set the partitions size
    """

    def _checkFormat(self, X):
        if X.shape[1] != 2:
            raise DataDimensionalityWarning("The input data must be in this form [[latitude, longitude]]")
        # Altri controlli?

    def __init__(self, precision=7, verbose=True):
        self.precision = precision
        self.verbose = verbose

    """
    l'output sarà una lista di encodings della forma (X.shape[0], 1)
    """

    def transform(self, X: np.ndarray):
        self._checkFormat(X)
        #encodes = np.chararray(X.shape[0], itemsize=self.precision)
        encodes = np.empty((X.shape[0]), dtype=F"U{self.precision}")

        if self.verbose: print(F"Encoding {X.shape[0]} points with precision {self.precision}", flush=True)


        for i, row in enumerate(tqdm(X, disable=not self.verbose, position=0, leave=True)):
            encodes[i] = geohash.encode(row[0], row[1], self.precision)

        return encodes
