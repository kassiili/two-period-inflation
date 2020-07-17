import unittest
import numpy as np
from astropy import units
import dataset_compute
import snapshot_obj


class TestDatasetCompute(unittest.TestCase):

    def setUp(self):
        self.snapID = 127
        self.simID = "V1_LR_fix"
        self.snapshot = snapshot_obj.Snapshot(self.simID,self.snapID)
        self.m31 = (1, 0)
        self.mw = (2, 0)

        cops = self.snapshot.get_subhalos("CentreOfPotential")
        self.dist_to_m31 = dataset_compute.distance_to_point(
            self.snapshot,
            cops[self.snapshot.index_of_halo(self.m31[0],self.m31[1])])
        self.dist_to_mw = dataset_compute.distance_to_point(
            self.snapshot,
            cops[self.snapshot.index_of_halo(self.mw[0],self.mw[1])])

        self.min_r = 0.01 * units.kpc.to(units.cm)
        self.max_r = 300 * units.kpc.to(units.cm)

    def test_split_satellites_by_distance(self):
        masks_sat, mask_isol = dataset_compute.mask_satellites_by_distance(
            self.snapshot, self.m31, self.mw)

        mask_sat_all = np.logical_or(
            dataset_compute.within_distance_range(self.dist_to_mw,
                                                  self.min_r,
                                                  self.max_r),
            dataset_compute.within_distance_range(self.dist_to_m31,
                                                  self.min_r,
                                                  self.max_r))

        # Check total number of satellites:
        n_sats1 = np.sum(np.logical_or(masks_sat[0], masks_sat[1]))
        n_sats2 = np.sum(mask_sat_all)
        self.assertEqual(n_sats1, n_sats2)

        # Check that there is no intersection between M31 and MW
        # satellites:
        self.assertFalse(np.all(np.logical_and(masks_sat[0],
                                               masks_sat[1])))

    def test_split_satellites_consistency_below_300kpc(self):
        masks_sat_gn, mask_isol_gn = \
            dataset_compute.mask_satellites_by_group_number(
                self.snapshot, self.m31, self.mw)
        masks_sat_r, mask_isol_r = \
            dataset_compute.mask_satellites_by_distance(
                self.snapshot, self.m31, self.mw)

        mask_below_r_m31 = (self.dist_to_m31 < self.max_r)
        mask_below_r_mw = (self.dist_to_mw < self.max_r)

        # Check that GN satellites within r are identified as R satellites:
        mask_test = np.logical_or(
            np.logical_and(masks_sat_gn[0], mask_below_r_m31),
            np.logical_and(masks_sat_gn[1], mask_below_r_mw)
        )
        self.assertEqual(np.sum(mask_test),
                         np.sum(np.logical_and(
                             mask_test, np.logical_or.reduce(masks_sat_r)))
                         )

    def test_split_satellites_by_distance_no_intersection(self):

        # Check that there is no intersection between isolated and
        # satellites:
        masks_sat, mask_isol = dataset_compute.mask_satellites_by_distance(
            self.snapshot, self.m31, self.mw)
        self.assertFalse(np.any(np.logical_and(masks_sat[0], mask_isol)))
        self.assertFalse(np.any(np.logical_and(masks_sat[1], mask_isol)))

        # No satellites beyond r:
        mask_above_r_m31 = (self.dist_to_m31 >= self.max_r)
        mask_above_r_mw = (self.dist_to_mw >= self.max_r)
        self.assertFalse(np.any(np.logical_or(
            np.logical_and(masks_sat[0], mask_above_r_m31),
            np.logical_and(masks_sat[1], mask_above_r_mw)
        )))

        # No isolated galaxies within r:
        mask_below_r_m31 = dataset_compute.within_distance_range(
            self.dist_to_m31, self.min_r, self.max_r)
        mask_below_r_mw = dataset_compute.within_distance_range(
            self.dist_to_mw, self.min_r, self.max_r)
        self.assertFalse(np.any(
            np.logical_and(mask_isol, np.logical_or(mask_below_r_m31,
                                                    mask_below_r_mw))
        ))

if __name__ == '__main__':
    unittest.main()
