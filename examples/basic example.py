from datetime import datetime
from Geolet.classifier.geoletclassifier import GeoletClassifier
from Geolet import distancers
from Geolet.classifier.geoletclassifier import prepare_y

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    df = pd.read_csv("animals_prepared.zip").sort_values(by=["tid", "t"])
    df = df[["tid", "class", "t", "c1", "c2"]]

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                 df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 test_size=.3,
                                                 stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 random_state=3)
    start = datetime.now()
    transform = GeoletClassifier(
        precision=3, # Geohash precision for the partitioning phase
        geolet_per_class=10,  # Number of candidate geolets to subsample randomly before the selecting phase
        selector='MutualInformation', # Name of the selector to use. Possible values are ["Random", "MutualInformation"]
        top_k=5,  # Top k geolets, according to the selector score, to use for transforming the entire dataset.
        trajectory_for_stats=100,  # Number of trajectory to subsample for selector scoring
        bestFittingMeasure=distancers.InterpolatedRouteDistance.interpolatedRootDistanceBestFitting, # best fitting measure to use
        distancer='IRD',  #Distance Measure to use for the final transformation. Possible values are ["E", "IRD"]
        verbose=True,
        n_jobs=4
    )

    X_train = df[df.tid.isin(tid_train)].drop(columns="class").values
    y_train = df[df.tid.isin(tid_train)].values[:, 1]

    X = df.drop(columns="class").values
    y = prepare_y(classes=df.values[:, 1], tids=df.values[:, 0])

    X_index, X_dist = transform.fit(X_train, y_train).transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_dist, y, test_size=.3, stratify=y, random_state=3)

    print((datetime.now() - start).total_seconds())

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)
