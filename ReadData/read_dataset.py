import numpy as np
import h5py
import os

# Numbers of files for different datasets:
#   - V1_LR_fix and V1_MR_fix: 16

def read_dataset(itype, att, nfiles=16, dataset='V1_LR_fix/snapshot_127_z000p000'):
    """ Read a selected dataset, itype is the PartType and att is the attribute name. """

    # Output array.
    data = []

    # Add relative path from current directory to the directory where the script is located:
    dirname = os.path.dirname(__file__)
    if not dirname:
        path = '../%s'%dataset 
    else:
        path = '%s/../%s'%(dirname,dataset)

    file_prefix = list(dataset.split("/")[-1])
    file_prefix = "".join(file_prefix[:4]) + "".join(file_prefix[8:])   # 'snapshot' -> 'snap'

    # Loop over each file and extract the data.
    for i in range(nfiles):
        filename = '%s/%s.%i.hdf5'%(path, file_prefix, i)
        f = h5py.File(filename, 'r')
        tmp = f['PartType%i/%s'%(itype, att)][...]
        data.append(tmp)

        # Get conversion factors.
        cgs     = f['PartType%i/%s'%(itype, att)].attrs.get('CGSConversionFactor')
        aexp    = f['PartType%i/%s'%(itype, att)].attrs.get('aexp-scale-exponent')
        hexp    = f['PartType%i/%s'%(itype, att)].attrs.get('h-scale-exponent')

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

#data = read_dataset(1, 'Coordinates')
#print(data[0,:])
