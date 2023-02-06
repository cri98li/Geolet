import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from Geolet.distancers.DistancerInterface import DistancerInterface
from Geolet.normalizers.NormalizerInterface import NormalizerInterface
from Geolet.normalizers.normalizer_utils import dataframe_pivot


class Euclidean_distancer(DistancerInterface):

    def __init__(self, normalizer, n_jobs=1, verbose=True):
        self.verbose = verbose
        self.normalizer = normalizer
        self.n_jobs = n_jobs

    def transform(self, tr_tid: np.ndarray, tr_time: np.ndarray, tr_X: np.ndarray, geo_tid: np.ndarray,
                  geo_time: np.ndarray, geo_X: np.ndarray):
        executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        processes = []
        for geolet_tid in tqdm(np.unique(geo_tid), disable=not self.verbose, position=0, leave=True):
            processes.append(executor.submit(self._computeDist, tr_tid, tr_time, tr_X,
                                             geo_time[geolet_tid == geo_tid],
                                             geo_X[geolet_tid == geo_tid]))

        distances = np.zeros((len(np.unique(tr_tid)), len(np.unique(geo_tid))))
        best_index = np.zeros((len(np.unique(tr_tid)), len(np.unique(geo_tid))))

        if self.verbose: print(f"Collecting distances from {len(processes)}")
        for i, process in enumerate(tqdm(processes)):
            ind, col = process.result()
            distances[:, i] = col
            best_index[:, i] = ind

        executor.shutdown(wait=True)

        return best_index, distances

    def _computeDist(self, tr_tid, tr_time, tr_X, geo_t, geo_X):
        unique_tr = np.unique(tr_tid)
        best_index = np.zeros(len(unique_tr))
        distances = np.zeros(len(unique_tr))

        for i, tid in enumerate(unique_tr):
            best_index[i], distances[i] = euclideanBestFitting(tr_time[tr_tid == tid], tr_X[tr_tid == tid], geo_t, geo_X,
                                                        normalizer=self.normalizer)
        return best_index, distances


#35s
def euclideanBestFitting(tr_time, tr_X, geo_time, geo_X, normalizer: NormalizerInterface, step=1):
    if len(geo_time) > len(tr_time):
        return euclideanBestFitting(geo_time, geo_X, tr_time, tr_X, normalizer)

    #tr_X = tr_X.copy()

    bestScore = math.inf
    best_i = -1
    for i in range(0, len(tr_time) - len(geo_time) + 1, step):
        tr_norm = normalizer._transformSingleTraj(tr_X[i:i + len(geo_time)])
        dist = ((tr_norm - geo_X) ** 2).sum()
        if dist < bestScore:
            bestScore = dist
            best_i = i

    return best_i, bestScore/len(geo_time)

#60s
"""def euclideanBestFitting(tr_time, tr_X, geo_time, geo_X, normalizer: NormalizerInterface, step=1):
    if len(geo_time) > len(tr_time):
        return euclideanBestFitting(geo_time, geo_X, tr_time, tr_X, normalizer)

    #tr_X = tr_X.copy()

    bestScore = math.inf
    best_i = -1
    for i in range(0, len(tr_time) - len(geo_time) + 1, step):
        #tr_norm = normalizer._transformSingleTraj(tr_X[i:i + len(geo_time)])
        dist=0
        for tr_el, geo_el in zip(tr_X[i:i + len(geo_time)], geo_X):
            dist = ((tr_el-geo_el-tr_X[i])**2).sum()
            if dist > bestScore:
                break
        if dist < bestScore:
            bestScore = dist
            best_i = i

    return best_i, bestScore/len(geo_time)"""
