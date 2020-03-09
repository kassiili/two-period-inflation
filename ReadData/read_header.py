import h5py
import os

def read_header(dataset='V1_LR_fix/snapshot_127_z000p000'):
    """ Read various attributes from the header group. """

    # Add relative path from current directory to the directory where the script is located:
    dirname = os.path.dirname(__file__)
    if not dirname:
        path = '../%s'%dataset 
    else:
        path = '%s/../%s'%(dirname,dataset)

    file_prefix = list(dataset.split("/")[-1])
    file_prefix = "".join(file_prefix[:4]) + "".join(file_prefix[8:])   # 'snapshot' -> 'snap'

    filename = '%s/%s.0.hdf5'%(path, file_prefix)
    f = h5py.File(filename, 'r')
    a       = f['Header'].attrs.get('Time')         # Scale factor.
    h       = f['Header'].attrs.get('HubbleParam')  # h.
    mass    = f['Header'].attrs.get('MassTable')    # 10^10 Msun
    boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].
    f.close()

    return a, h, mass, boxsize

#a, h, mass, boxsize = read_header()
#print(mass)

