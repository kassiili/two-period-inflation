import numpy as np
import h5py
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snapshot_obj import Snapshot

# Read data:
simID="CDM_V1_LR"; snapID=127; partt=5
snap = Snapshot(simID,snapID)
gns = snap.get_subhalos('GroupNumber')
sgns = snap.get_subhalos('SubGroupNumber')
ids = snap.get_subhalos_IDs(part_type=[partt])
all_ids = snap.get_subhalos_IDs()

# Select halo:
gn = 1; sgn = 0
mask = np.logical_and(gns==gn, sgns==sgn)
ids = np.array(ids[mask][0])
mask = np.logical_and(gns==gn, sgns==sgn)
all_ids = np.array(all_ids[mask][0])

np.savetxt("ids_{}-{}_group-{}-{}_type{}.dat"\
        .format(simID,snapID,gn,sgn,partt), ids, fmt='%d')

np.savetxt("ids_{}-{}_group-{}-{}_all.dat"\
        .format(simID,snapID,gn,sgn), all_ids, fmt='%d')
