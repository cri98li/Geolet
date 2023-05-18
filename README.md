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
#TODO
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

//TODO
 
