from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import trajectory_toolkit
from trajectory_toolkit.geolet.Geolet import Geolet, prepare_y

if __name__ == '__main__':
    df = pd.read_csv("../trajectory_toolkit/test/datasets/animals.zip").sort_values(by=["tid", "t"])
    df = df[["tid", "class", "t", "c1", "c2"]]

    tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                                 df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 test_size=.3,
                                                 stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                                 random_state=3)
    start = datetime.now()
    transform = Geolet(precision=3, geolet_per_class=100000, selector='MutualInformation', top_k=50,
                       trajectory_for_stats=10000000,
                       bestFittingMeasure=trajectory_toolkit.distancers.InterpolatedRouteDistance.interpolatedRootDistanceBestFitting,
                       distancer='IRD',
                       verbose=True, n_jobs=12)

    X_train = df[df.tid.isin(tid_train)].drop(columns="class").values
    y_train = df[df.tid.isin(tid_train)].values[:, 1]

    X = df.drop(columns="class").values
    y = prepare_y(classes=df.values[:, 1], tids=df.values[:, 0])

    # Convert classes to int (XGBoost doesn't support char as class)
    mapping = {'D': 0, "E": 1, "C": 2}
    conversion_f = np.vectorize(lambda x: mapping[x[0]])
    y = conversion_f(y)

    X_index, X_dist = transform.fit(X_train, y_train).transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_dist, y, test_size=.3, stratify=y, random_state=3)

    print((datetime.now() - start).total_seconds())

    bst = XGBClassifier(n_estimators=10)
    bst.fit(X_train, y_train)

    y_pred = bst.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)
