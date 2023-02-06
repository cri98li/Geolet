import unittest
import random

import numpy as np

from Geolet.partitioners.Geohash import Geohash

class TestAlgorithms(unittest.TestCase):
    def test_geohash_partitioner(self):
        array = np.array([
            [28.6132, 77.2291]
        ])

        part = Geohash(precision=7)

        self.assertTrue(np.all(part.transform(array) == np.array(["ttnfv2u"])))



if __name__ == "__main__":
    unittest.main()
