import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.autonotebook import tqdm

from Geolet.distancers.DistancerInterface import DistancerInterface
from Geolet.normalizers.NormalizerInterface import NormalizerInterface


class InterpolatedRootDistance(DistancerInterface):

    def __init__(self, normalizer, n_jobs=1, verbose=True):
        self.verbose = verbose
        self.normalizer = normalizer
        self.n_jobs = n_jobs

    # trajectories = tid, class, time, c1, c2
    # restituisce nparray con pos0= cluster e poi
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
            best_index[i], distances[i] = interpolatedRootDistanceBestFitting(tr_time[tr_tid == tid],
                                                                              tr_X[tr_tid == tid], geo_t, geo_X,
                                                                              normalizer=self.normalizer)
        return best_index, distances


def interpolatedRootDistanceBestFitting(tr_time, tr_X, geo_time, geo_X, normalizer: NormalizerInterface,
                                        step=1):  # nan == end
    if len(geo_time) > len(tr_time):
        return interpolatedRootDistanceBestFitting(geo_time, geo_X, tr_time, tr_X, normalizer)

    tr_X = np.hstack((tr_X, tr_time.reshape((len(tr_time), 1))))
    geo_X = np.hstack((geo_X, geo_time.reshape((len(geo_time), 1))))

    bestScore = math.inf
    best_i = -1
    for i in range(0, len(tr_time) - len(geo_time) + 1, step):
        tr_norm = normalizer._transformSingleTraj(tr_X[i:i + len(geo_time)])
        dist = trajectory_distance(tr_norm, geo_X)
        if dist < bestScore:
            bestScore = dist
            best_i = i

    return best_i, bestScore/len(geo_time)


# From here credits to Riccardo Guidotti
def spherical_distance(a, b):
    lat1 = a[1]
    lon1 = a[0]
    lat2 = b[1]
    lon2 = b[0]
    R = 6371000
    rlon1 = lon1 * math.pi / 180.0
    rlon2 = lon2 * math.pi / 180.0
    rlat1 = lat1 * math.pi / 180.0
    rlat2 = lat2 * math.pi / 180.0
    dlon = (rlon1 - rlon2) / 2.0
    dlat = (rlat1 - rlat2) / 2.0
    lat12 = (rlat1 + rlat2) / 2.0
    sindlat = math.sin(dlat)
    sindlon = math.sin(dlon)
    cosdlon = math.cos(dlon)
    coslat12 = math.cos(lat12)
    f = sindlat * sindlat * cosdlon * cosdlon + sindlon * sindlon * coslat12 * coslat12
    f = math.sqrt(f)
    f = math.asin(f) * 2.0  # the angle between the points
    f *= R
    return f


def trajectory_distance(tr1, tr2):
    if len(tr1[0]) == 1 and len(tr2[0]) == 1:
        return .0

    i1 = 0
    i2 = 0
    np = 0

    last_tr1 = tr1[i1]
    last_tr2 = tr2[i2]

    dist = spherical_distance(last_tr1, last_tr2)
    np += 1

    while True:
        if i1 >= (len(tr1) - 1) or i2 >= (len(tr2) - 1):
            break

        step_tr1 = spherical_distance(last_tr1, tr1[i1 + 1])
        step_tr2 = spherical_distance(last_tr2, tr2[i2 + 1])

        if step_tr1 < step_tr2:
            i1 += 1
            last_tr1 = tr1[i1]
            last_tr2 = closest_point_on_segment(last_tr2, tr2[i2 + 1], last_tr1)
        elif step_tr1 > step_tr2:
            i2 += 1
            last_tr2 = tr2[i2]
            last_tr1 = closest_point_on_segment(last_tr1, tr1[i1 + 1], last_tr2)
        else:
            i1 += 1
            i2 += 1
            last_tr1 = tr1[i1]
            last_tr2 = tr2[i2]

        d = spherical_distance(last_tr1, last_tr2)

        dist += d
        np += 1

    for i in range(i1, len(tr1)):
        d = spherical_distance(tr2[-1], tr1[i])
        dist += d
        np += 1

    for i in range(i2, len(tr2)):
        d = spherical_distance(tr1[-1], tr2[i])
        dist += d
        np += 1

    dist = 1.0 * dist / np

    return dist


def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    sz1 = a[2]
    sz2 = b[2]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1
    z_delta = sz2 - sz1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
    if u < 0:
        closest_point = a
    elif u > 1:
        closest_point = b
    else:
        cp_x = sx1 + u * x_delta
        cp_y = sy1 + u * y_delta
        dist_a_cp = spherical_distance(a, [cp_x, cp_y, 0])
        if dist_a_cp != 0:
            cp_z = sz1 + z_delta / (spherical_distance(a, b) / spherical_distance(a, [cp_x, cp_y, 0]))
        else:
            cp_z = a[2]
        closest_point = [cp_x, cp_y, cp_z]

    return closest_point
