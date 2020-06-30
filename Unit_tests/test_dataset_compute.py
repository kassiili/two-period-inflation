import unittest
import numpy as np
import dataset_compute
import snapshot_obj


class TestDatasetCompute(unittest.TestCase):

    def test_compute_cumulative_mass(self):
        sim_id = "V1_LR_fix"
        snap = snapshot_obj.Snapshot(sim_id, 127)
        check = dataset_compute.compute_mass_accumulation(snap)
        dataset = 'GroupNumber'
        #check = dataset_compute.group_particles_by_subhalo(snap, dataset,
        #        part_type=[1])
        #test_gns = np.all([np.all(a == a[0]) for a in check['gns']])
        self.assertEqual(False, True)


if __name__ == '__main__':
    unittest.main()
