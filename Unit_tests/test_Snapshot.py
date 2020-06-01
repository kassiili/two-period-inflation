import unittest as ut
import numpy as np
import h5py
import sys, os

from snapshot_obj import Snapshot
import data_file_manipulation

class TestSnapshot(ut.TestCase):

    def setUp(self):
        self.snapID = 127
        self.simID = "CDM_V1_LR"
        self.snapshot = Snapshot(self.simID,self.snapID)
        self.grp_path = data_file_manipulation.get_data_path('group',\
                self.simID, self.snapID)
        self.part_path = data_file_manipulation.get_data_path('part',\
                self.simID, self.snapID)

    def test_get_particles(self):
        attr = "Coordinates"
        pt = 1
        data = self.snapshot.get_particles(attr,part_type=[pt])

        with h5py.File(self.snapshot.part_file,'r') as f:
            part_n = f['link1/Header'].attrs.get('NumPart_Total')

        self.assertEqual(part_n[pt], len(data), \
                ("Size of dataset {} does not match that " +\
                "listed in the header for particle type {}")\
                .format(attr,pt))

if __name__ == '__main__':
    ut.main()
