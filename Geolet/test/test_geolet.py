import unittest
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

import Geolet
from Geolet.classifier.geoletclassifier import GeoletClassifier


class TestAlgorithms(unittest.TestCase):
    def test_geolet_Random_animals(self):
        df = pd.read_csv("datasets/animals.zip").sort_values(by=["tid", "t"])
        df = df[["tid", "class", "t", "c1", "c2"]]

        tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                     df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     test_size=.3,
                                                     stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     random_state=3)

        transform = GeoletClassifier(precision=10, top_k=10)

        X = df.drop(columns="class").values
        y = df.values[:, 1]

        transform.fit_transform(X, y)

    def test_geolet_MutualInformation_animals(self):
        df = pd.read_csv("datasets/animals.zip").sort_values(by=["tid", "t"])
        df = df[["tid", "class", "t", "c1", "c2"]]

        tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                     df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     test_size=.3,
                                                     stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     random_state=3)

        transform = GeoletClassifier(precision=6, geolet_per_class=10, selector='MutualInformation',
                                     bestFittingMeasure=Geolet.distancers.Euclidean.euclideanBestFitting,
                                     )

        X = df.drop(columns="class").values
        y = df.values[:, 1]

        transform.fit_transform(X, y)

    def test_eliminami(self):
        df = pd.read_csv("datasets/animals.zip").sort_values(by=["tid", "t"])
        df = df[["tid", "class", "t", "c1", "c2"]]

        tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                     df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     test_size=.3,
                                                     stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                     random_state=3)
        start = datetime.now()
        transform = GeoletClassifier(precision=3, geolet_per_class=100000, selector='MutualInformation', top_k=50,
                                     trajectory_for_stats=10000000,
                                     bestFittingMeasure=Geolet.distancers.Euclidean.euclideanBestFitting,
                                     verbose=True, n_jobs=12)

        X = df.drop(columns="class").values
        y = df.values[:, 1]

        transform.fit_transform(X, y)

        print((start - datetime.now()).total_seconds() * 1000)


    def test_eliminami2(self):
        df = pd.read_csv("datasets/vehicles.zip").sort_values(by=["tid", "t"])
        df = df[["tid", "class", "t", "c1", "c2"]]

        df["c1"] = df.c1 / 100000
        df["c2"] = df.c2 / 100000
        for _ in range(5):

            tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                         df.groupby(by=["tid"]).max().reset_index()["class"],
                                                         test_size=.3,
                                                         stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                         random_state=3)
            start = datetime.now()
            transform = GeoletClassifier(precision=6, geolet_per_class=150, selector='MutualInformation', top_k=10,
                                         trajectory_for_stats=100,
                                         bestFittingMeasure=Geolet.distancers.InterpolatedRouteDistance.interpolatedRootDistanceBestFitting,
                                         distancer='IRD',
                                         verbose=True, n_jobs=12)

            X = df.drop(columns="class").values
            y = df.values[:, 1]

            transform.fit_transform(X, y)

            print((start - datetime.now()).total_seconds()*1000)





if __name__ == "__main__":
    unittest.main()
