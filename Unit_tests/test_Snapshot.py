import unittest as ut
import numpy as np
import h5py

from snapshot_obj import Snapshot
import data_file_manipulation

class TestSnapshot(ut.TestCase):

    def setUp(self):
        self.snapID = 127
        self.simID = "V1_LR_fix"
        self.snapshot = Snapshot(self.simID,self.snapID)
        self.grp_path = data_file_manipulation.get_data_path('group',\
                self.simID, self.snapID)
        self.part_path = data_file_manipulation.get_data_path('part',\
                self.simID, self.snapID)

    def test_get_particles_size(self):
        # Check: attr dataset is of right size:
        attr = "ParticleIDs"
        with h5py.File(self.snapshot.part_file,'r') as f:
            part_n = f['link1/Header'].attrs.get('NumPart_Total')
    
        pts = [[0,1,4,5],[0,1,2,3,4,5],[0],[1]]
        for pt in pts:
            data = self.snapshot.get_particles(attr,part_type=pt)
    
            self.assertEqual(np.sum(part_n[pt]), data.size, \
                    ("Size of dataset {} does not match that " +\
                    "listed in the header for particle type {}")\
                    .format(attr,pt))

    def test_get_subhalos_IDs_halocount(self):
        pt = 1
        IDs = self.snapshot.get_subhalos_IDs(pt)
        gns = self.snapshot.get_subhalos("GroupNumber")
        sgns = self.snapshot.get_subhalos("SubGroupNumber")
        self.assertEqual(IDs.size, gns.size, "Halo number incorrect")

    def test_get_all_bound_IDs_rightOnes(self):
        test = self.snapshot.get_bound_particles("ParticleID")
        
        pts = [0,1,2,3,4,5]
        part_gns = self.snapshot.get_particles("GroupNumber",\
                part_type=pts)
        part_sgns = self.snapshot.get_particles("SubGroupNumber",\
                part_type=pts)
        part_IDs = self.snapshot.get_particles("ParticleIDs",\
                part_type=pts)

        # Select bound particles:
        bound_IDs = part_IDs[np.logical_and(\
                part_gns>0,part_gns<100000)]

        self.assertTrue(np.in1d(test,bound_IDs).all(),\
                "Retrieved IDs do not match bound IDs from particle data")

    def test_get_subhalos_IDs_rightOnes(self):
        pt = 1

        IDs = self.snapshot.get_subhalos_IDs(pt)
        gns = self.snapshot.get_subhalos("GroupNumber")
        sgns = self.snapshot.get_subhalos("SubGroupNumber")

        part_gns = self.snapshot.get_particles("GroupNumber",
                part_type=[pt])
        part_sgns = self.snapshot.get_particles("SubGroupNumber",
                part_type=[pt])
        part_IDs = self.snapshot.get_particles("ParticleIDs",\
                part_type=[pt])

        check_idx = np.arange(gns.size)[sgns==0][:2] # M31 and MW
        check_idx = np.append(check_idx,[100,250])   # some random halos
        for i in check_idx:
            halo_IDs = part_IDs[np.logical_and(\
                    part_gns==gns[i],part_sgns==sgns[i])]
            self.assertTrue(np.in1d(IDs[i],halo_IDs).all(),\
                    ("Retrieved bound particles of type {} not right " +\
                    "for halo ({},{})").format(pt,gns[i],sgns[i]))

    def test_get_subhalos_IDs_order(self):
        # Order should be from most bound to least bound
        pt = 1

        IDs = self.snapshot.get_subhalos_IDs(pt)
        gns = self.snapshot.get_subhalos("GroupNumber")
        sgns = self.snapshot.get_subhalos("SubGroupNumber")

        bound_IDs = self.snapshot.get_bound_particles("ParticleID")
        bindErg = self.snapshot.\
                get_bound_particles("Particle_Binding_Energy")

        check_idx = np.arange(gns.size)[sgns==0][:2] # M31 and MW
        check_idx = np.append(check_idx,[920,71])   # some random halos
        for i in check_idx:
            # Get binding energies of particle in the same order as in 
            # IDs[i] and check that they are in ascending order:
            idxs = np.in1d(bound_IDs, IDs[i])
            sort_halo = np.argsort(bound_IDs[idxs])
            invsort_halo = np.argsort(np.argsort(IDs[i]))
            self.assertTrue(\
                    (np.diff(bindErg[sort_halo][invsort_halo])>=0).all())

    def test_get_attribute(self):
        z = self.snapshot.get_attribute('Redshift', 'Header')
        self.assertTrue(isinstance(z, float))

    def test_get_halo_number(self):
        print(self.snapshot.get_halo_number([]))
        print(self.snapshot.get_halo_number([1]))
        self.assertTrue(True)

if __name__ == '__main__':
    ut.main()
