# Geolet - Interpretable GPS Trajectory Classifier

Researchers, businesses, and governments use mobility data to make decisions that affect people's lives in many ways, employing accurate but opaque deep learning models that are difficult to interpret from a human standpoint. 
To address these limitations, we propose Geolet, a human-interpretable machine-learning model for trajectory classification. 
We use discriminative sub-trajectories extracted from mobility data to turn trajectories into a simplified representation that can be used as input by any machine learning classifier. 


## Setup

### Using PyPI

```bash
  pip install geolet
```

### Manual Setup

```bash
git clone https://github.com/cri98li/Geolet
cd Geolet
pip install -e .
```

Dependencies are listed in `requirements.txt`.


## Running the code

```python
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from Geolet.classifier.geoletclassifier import GeoletClassifier, prepare_y
from Geolet import distancers


df = pd.read_csv("animals_prepared.zip").sort_values(by=["tid", "t"])
df = df[["tid", "class", "t", "c1", "c2"]]

tid_train, tid_test, _, _ = train_test_split(df.groupby(by=["tid"]).max().reset_index()["tid"],
                                             df.groupby(by=["tid"]).max().reset_index()["class"],
                                             test_size=.3,
                                             stratify=df.groupby(by=["tid"]).max().reset_index()["class"],
                                             random_state=3)
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
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

Jupyter notebooks with examples on real datasets can be found in the `examples/` directory.


## Docs and reference


You can find the software documentation in the `/docs/` folder and 
a powerpoint presentation on Geolet can be found [here](http://example.org).
You can cite this work with
```
@inproceedings{DBLP:conf/ida/LandiSGMN23,
  author       = {Cristiano Landi and
                  Francesco Spinnato and
                  Riccardo Guidotti and
                  Anna Monreale and
                  Mirco Nanni},
  title        = {Geolet: An Interpretable Model for Trajectory Classification},
  booktitle    = {{IDA}},
  series       = {Lecture Notes in Computer Science},
  volume       = {13876},
  pages        = {236--248},
  publisher    = {Springer},
  year         = {2023}
}
```


## Extending the algorithm

The original Geolet code, i.e., the code used for the experiments in the paper, is available in the /original_code branch.

The code in the main branch is a reimplementation that speeds up the execution time by about 7%.
 
