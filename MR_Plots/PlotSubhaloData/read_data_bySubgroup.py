import numpy as np
import h5py

def read_partIDs(nfiles=192):
    """ Read particle IDs of each subhalo into a dictionary.  """

    path = '/home/kassiili/SummerProject/practise-with-datasets/V1_MR_fix/groups_127_z000p000'

    # Output dictionary (keys are subgroup numbers and values arrays of particle IDs).
    data = dict()

    # Loop over each file and extract the data.
    for i in range(nfiles):
        part_type = 1
        f = h5py.File('%s/eagle_subfind_tab_127_z000p000.%i.hdf5'%(path, i), 'r')
        IDs = f['IDs/ParticleID'][...]
        subOffsets = f['Subhalo/SubOffset'][...]
        subLengthsType = f['Subhalo/SubLengthType'][...]
        subGroupNumbers = f['Subhalo/SubGroupNumber'][...]

#        for idx, subGroupNumber in enumerate(subGroupNumbers):
#            print(subGroupNumber)
#            start = np.asscalar(subOffsets[idx] + np.sum(subLengthsType[:part_type]))
#            end = start + np.asscalar(subLengthsType[idx, part_type + 1])
#
#            data[subGroupNumber] = IDs[start:end]

        f.close()

    return data

