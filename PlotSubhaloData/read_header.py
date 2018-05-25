import h5py

def read_header():
    """ Read various attributes from the header group. """
    path    = '/home/kassiili/SummerProject/DatasetPractise/V1_LR_fix/snapshot_127_z000p000'
    f       = h5py.File('%s/snap_127_z000p000.0.hdf5'%path, 'r')
    a       = f['Header'].attrs.get('Time')         # Scale factor.
    h       = f['Header'].attrs.get('HubbleParam')  # h.
    mass    = f['Header'].attrs.get('MassTable')    # particle mass
    boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].
    f.close()

    return a, h, mass, boxsize
