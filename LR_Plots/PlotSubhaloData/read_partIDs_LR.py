import numpy as np
import h5py

def read_partIDs(att, nfiles=95):
    """ Read a selected dataset from the IDs group of the group data files, att is the attribute name.  """

    path = '/home/kassiili/SummerProject/practise-with-datasets/V1_LR_fix/groups_127_z000p000'

    # Output array.
    data = []

    # Loop over each file and extract the data.
    for i in range(nfiles):
        f = h5py.File('%s/eagle_subfind_tab_127_z000p000.%i.hdf5'%(path, i), 'r')
        tmp = f['IDs/%s'%att][...]
        data.append(tmp)

        # Get conversion factors.
        cgs     = f['IDs/%s'%att].attrs.get('CGSConversionFactor')
        aexp    = f['IDs/%s'%att].attrs.get('aexp-scale-exponent')
        hexp    = f['IDs/%s'%att].attrs.get('h-scale-exponent')

        # Get expansion factor and Hubble parameter from the header.
        a       = f['Header'].attrs.get('Time')
        h       = f['Header'].attrs.get('HubbleParam')

        f.close()

    # Combine to a single array.
    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)

    # Convert to physical.
#    if data.dtype != np.int32 and data.dtype != np.int64:
#        data = np.multiply(data, cgs * a**aexp * h**hexp, dtype='f8')

    return data

