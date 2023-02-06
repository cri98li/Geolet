import unittest
import random

import numpy as np

from Geolet.normalizers.FirstPoint import FirstPoint

class TestAlgorithms(unittest.TestCase):
    def test_firstpoint_normalizer(self):
        norm = FirstPoint()
        X = np.array([
            [1, 2],
            [2, 2],
            [7, 2],
        ], dtype='O')
        part = np.array(["ciao", "ciao", "addio"])

        X_expected = np.array([
            [0, 0],
            [1, 0],
            [0, 0]
        ], dtype='O')

        self.assertTrue(np.all(X_expected == norm.transform(X=X, part=part)))


if __name__ == "__main__":
    unittest.main()
