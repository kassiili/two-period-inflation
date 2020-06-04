import unittest
import numpy as np

from subhalo import Subhalo
from snapshot_obj import Snapshot


class TestSubhalo(unittest.TestCase):

    def setUp(self):
        self.snap_id = 127
        self.gn = 1.0
        self.sgn = 1.0
        self.halo = Subhalo("CDM_V1_LR", self.snap_id, self.gn, self.sgn)
        self.snapshot = Snapshot("CDM_V1_LR", 127)

    def test_get_dataset(self):
        dataset = "CentreOfPotential"
        test = self.halo.get_halo_data(dataset, self.snap_id)
        gns = self.snapshot.get_subhalos("GroupNumber")
        sgns = self.snapshot.get_subhalos("SubGroupNumber")
        data_snap = self.snapshot.get_subhalos(dataset)
        correct = data_snap[
            np.logical_and(self.gn == gns, self.sgn == sgns)][0]

        self.assertTrue(np.array_equal(test, correct),
                         "Retrieved halo dataset {} not correct".format(
                             dataset))

    def test_distance_to_central(self):
        test = self.halo.distance_to_central(self.snap_id)
        print(test)
        self.assertTrue(isinstance(test, np.ndarray))
        self.assertEqual(test.shape, (3,))
        self.assertTrue(isinstance(test[0], float))

    def test_get_ids_type(self):
        test = self.halo.get_ids(self.snap_id)
        self.assertTrue(isinstance(test, np.ndarray))
        self.assertTrue(isinstance(test[0], float))

    def test_trace(self):
        test = self.halo.trace(stop=125)
        print(test)
        self.assertTrue(isinstance(test, dict))

if __name__ == '__main__':
    unittest.main()
