import h5py

def read_header(dataset='LR'):
    """ Read various attributes from the header group. Optional argument determines, which dataset is used (Low-Res or Medium-Res). """
    path    = '/home/kassiili/SummerProject/practise-with-datasets/V1_%s_fix/snapshot_127_z000p000'%dataset
    f       = h5py.File('%s/snap_127_z000p000.0.hdf5'%path, 'r')
    a       = f['Header'].attrs.get('Time')         # Scale factor.
    h       = f['Header'].attrs.get('HubbleParam')  # h.
    mass    = f['Header'].attrs.get('MassTable')    # 10^10 Msun
    boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].
    f.close()

    return a, h, mass, boxsize
