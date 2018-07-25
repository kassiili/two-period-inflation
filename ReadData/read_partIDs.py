import numpy as np
import h5py
import os

def read_partIDs(att, nfiles=95, dataset='V1_LR_fix/groups_127_z000p000'):
    """ Read a selected dataset from the IDs group of the group data files, att is the attribute name.  """

    # Add relative path from current directory to the directory where the script is located:
    dirname = os.path.dirname(__file__)
    if not dirname:
        path = '../%s'%dataset 
    else:
        path = '%s/../%s'%(dirname,dataset)

    file_prefix = list(dataset.split("/")[-1])
    file_prefix = "eagle_subfind_tab" + "".join(file_prefix[6:])

    # Output array.
    data = []

    # Loop over each file and extract the data.
    for i in range(nfiles):
        filename = '%s/%s.%i.hdf5'%(path, file_prefix, i)
        f = h5py.File(filename, 'r')
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
    if data.dtype != np.int32 and data.dtype != np.int64:
        data = np.multiply(data, cgs * a**aexp * h**hexp, dtype='f8')

    return data

#data = read_partIDs('ParticleID')
#print(data[0])
