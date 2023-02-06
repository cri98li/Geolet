import numpy as np
from sklearn.base import TransformerMixin

from Geolet.partitioners.PartitionerInterface import PartitionerInterface
from Geolet.partitioners.Geohash import Geohash

from Geolet.normalizers.NormalizerInterface import NormalizerInterface
from Geolet.normalizers.FirstPoint import FirstPoint

from Geolet.selectors.SelectorInterface import SelectorInterface
from Geolet.selectors.Random import Random
from Geolet.selectors.RandomInformationGain import RandomInformationGain

from Geolet.distancers.DistancerInterface import DistancerInterface
from Geolet.distancers.Euclidean import Euclidean_distancer
from Geolet.distancers.InterpolatedRouteDistance import InterpolatedRootDistance


class GeoletClassifier(TransformerMixin):
    def __init__(self, precision=7,
                 partitioner='Geohash',
                 normalizer='FirstPoint',
                 selector='Random', geolet_per_class=10, bestFittingMeasure=lambda x: x, top_k=3,
                 trajectory_for_stats=100, n_neighbors=3,
                 distancer='E',
                 n_jobs=1,
                 random_state=42,
                 verbose=False):

        self.verbose = verbose

        if partitioner == 'Geohash':
            self.partitioner = Geohash(precision=precision, verbose=verbose)
        elif isinstance(partitioner, PartitionerInterface):
            self.partitioner = partitioner
        else:
            raise ValueError(
                f"partitioner={partitioner} unsupported. You can use a custom partitioner by passing its instance")

        if normalizer == 'FirstPoint':
            self.normalizer = FirstPoint()
        elif isinstance(normalizer, NormalizerInterface):
            self.normalizer = normalizer
        else:
            raise ValueError(
                f"normalizer={normalizer} unsupported. You can use a custom normalizer by passing its instance")

        if selector == 'Random':
            self.selector = Random(normalizer=self.normalizer, n_geolet_per_class=top_k, verbose=verbose)
        elif selector == 'MutualInformation':
            self.selector = RandomInformationGain(normalizer=self.normalizer, bestFittingMeasure=bestFittingMeasure,
                                                  top_k=top_k, n_geolet_per_class=geolet_per_class,
                                                  estimation_trajectories_per_class=trajectory_for_stats,
                                                  n_neighbors=n_neighbors, n_jobs=n_jobs, random_state=random_state,
                                                  verbose=verbose)
        elif isinstance(selector, SelectorInterface):
            self.selector = selector
        else:
            raise ValueError(
                f"selector={selector} unsupported. You can use a custom selector by passing its instance")

        if distancer == 'E':
            self.distancer = Euclidean_distancer(normalizer=self.normalizer, n_jobs=n_jobs, verbose=verbose)
        elif distancer == 'IRD':
            self.distancer = InterpolatedRootDistance(normalizer=self.normalizer, n_jobs=n_jobs, verbose=verbose)
        elif isinstance(distancer, DistancerInterface):
            self.distancer = distancer
        else:
            raise ValueError(
                f"distancer={distancer} unsupported. You can use a custom distancer by passing its instance")

    def fit(self, X: np.ndarray, y: np.ndarray):
        tid = X[:, 0]
        time = X[:, 1]
        lat_lon = X[:, 2:]

        partitions = self.partitioner.transform(lat_lon)

        self.sel_tid, sel_y, self.sel_time, self.lat_lon = self.selector.transform(tid, y, time, lat_lon, partitions)

        return self

    # specific order: tid, class, time, lat, lon
    def transform(self, X: np.ndarray):
        tid = X[:, 0]
        time = X[:, 1]
        lat_lon = X[:, 2:]

        return self.distancer.transform(tid, time, lat_lon, self.sel_tid, self.sel_time, self.lat_lon)

def prepare_y(classes, tids):
    selected_tr_classes = [classes[i] for i in range(len(tids) - 1) if tids[i] != tids[i + 1]]
    selected_tr_classes.append(classes[-1:])
    return selected_tr_classes