import multiprocessing

from concurrent.futures import ProcessPoolExecutor
import random
from itertools import groupby

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from tqdm.autonotebook import tqdm

from Geolet.selectors.Random import Random
from Geolet.selectors.SelectorInterface import SelectorInterface


class RandomInformationGain(SelectorInterface):
    def __init__(self, normalizer, bestFittingMeasure, top_k=10, n_geolet_per_class=100, estimation_trajectories_per_class=10,
                 n_neighbors=3, n_jobs=1, random_state=None, verbose=True):
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_geolet_per_class = n_geolet_per_class
        self.n_trajectories = estimation_trajectories_per_class
        self.top_k = top_k
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.normalizer = normalizer
        self.bestFittingMeasure = bestFittingMeasure

    """
    l'output sarà:
    tid, class, time, c1, c2, geohash
    """

    def transform(self, tid: np.ndarray, classes: np.ndarray, time: np.ndarray, X: np.ndarray, partid: np.ndarray):

        geolets_tid, geolets_classes, geolets_time, geolets_X = Random(self.normalizer, n_geolets=self.n_geolet_per_class) \
            .transform(tid, classes, time, X, partid)

        #selected_tr_tid = selected_tr_tid[np.invert(np.isin(selected_tr_tid, geolets_tid))]

        unique_classes = np.unique(classes)
        selected_rows = np.full(len(tid), False)
        n = 0
        for classe in unique_classes:
            tmp = np.vstack((tid, classes)).T
            selected_tr_tid = np.unique(tmp[tmp[:, 1]==classe][:, 0])
            n += min(self.n_geolet_per_class, len(selected_tr_tid))
            selected_rows |= np.isin(tid, random.sample(selected_tr_tid.tolist(), min(self.n_geolet_per_class, len(selected_tr_tid))))
        selected_tr_tid = tid[selected_rows]
        selected_tr_time = time[selected_rows]
        selected_tr_X = X[selected_rows]

        dist_matrix = np.zeros((n, len(np.unique(geolets_tid))))

        if self.verbose: print(F"Computing scores")

        executor = ProcessPoolExecutor(max_workers=self.n_jobs)
        processes = []
        for geolet_tid in tqdm(np.unique(geolets_tid), disable=not self.verbose, position=0, leave=True):
            processes.append(executor.submit(self._computeDist, selected_tr_tid, selected_tr_time, selected_tr_X,
                                             geolets_time[geolet_tid == geolets_tid],
                                             geolets_X[geolet_tid == geolets_tid]))

        for i, process in enumerate(tqdm(processes, disable=not self.verbose, position=0, leave=True)):
            res = process.result()
            dist_matrix[:,i] = res

        selected_tr_classes = [classes[selected_rows][i] for i in range(len(selected_tr_tid)-1) if selected_tr_tid[i] != selected_tr_tid[i+1]]
        selected_tr_classes.append(classes[selected_rows][-1:])

        mutualInfo = mutual_info_classif(dist_matrix, selected_tr_classes, #[k for k, g in groupby(classes[selected_rows])],
                                         n_neighbors=self.n_neighbors, random_state=self.random_state)

        ri_selected_tid = []
        for i, (score, geolet_tid) in enumerate(
                sorted(zip(mutualInfo, np.unique(geolets_tid)), key=lambda x: x[0], reverse=True)):
            if i >= self.top_k:
                break
            ri_selected_tid.append(geolet_tid)
            if self.verbose:
                print(F"{i}.\t score={score}")

        to_keep_indeces = np.isin(geolets_tid, ri_selected_tid)

        return geolets_tid[to_keep_indeces], geolets_classes[to_keep_indeces], geolets_time[to_keep_indeces], \
               geolets_X[to_keep_indeces]

    def _computeDist(self, tr_tid, tr_time, tr_X, geo_t, geo_X):
        unique_tr = np.unique(tr_tid)
        distances = np.zeros(len(unique_tr))

        for i, tid in enumerate(unique_tr):
            best_index, distance = self.bestFittingMeasure(tr_time[tr_tid == tid], tr_X[tr_tid == tid], geo_t, geo_X,
                                                           normalizer=self.normalizer)
            distances[i] = distance
        return distances
