import unittest
import numpy as np

from snapshot_obj import Snapshot
import trace_halo


class TestSubhalo(unittest.TestCase):

    def setUp(self):
        self.snap_id_ref = 127
        self.snap_id_exp = 126
        self.snap_ref = Snapshot("CDM_V1_LR", self.snap_id_ref)
        self.snap_exp = Snapshot("CDM_V1_LR", self.snap_id_exp)

    def test_identify_group_numbers(self):
        gns_ref = self.snap_ref.get_subhalos("GroupNumber")
        gns_exp = self.snap_exp.get_subhalos("GroupNumber")
        out = trace_halo.identify_group_numbers(gns_ref, gns_exp)
        test = [isinstance(out[i], np.int_) for i in range(out.size)]
        self.assertTrue(all(test))


if __name__ == '__main__':
    unittest.main()
