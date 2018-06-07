import numpy as np
import h5py

def read_dataset_dm_mass():
    """ Special case for the mass of dark matter particles. """
    path = '/home/kassiili/SummerProject/practise-with-datasets/V1_LR_fix/snapshot_127_z000p000'
    f           = h5py.File('%s/snap_127_z000p000.0.hdf5'%path, 'r')
    h           = f['Header'].attrs.get('HubbleParam')
    a           = f['Header'].attrs.get('Time')
    dm_mass     = f['Header'].attrs.get('MassTable')[1]
    n_particles = f['Header'].attrs.get('NumPart_Total')[1]

    # Create an array of length n_particles each set to dm_mass.
    m = np.ones(n_particles, dtype='f8') * dm_mass 

    # Use the conversion factors from the mass entry in the gas particles.
    cgs  = f['PartType0/Masses'].attrs.get('CGSConversionFactor')
    aexp = f['PartType0/Masses'].attrs.get('aexp-scale-exponent')
    hexp = f['PartType0/Masses'].attrs.get('h-scale-exponent')
    f.close()

    # Convert to physical.
    m = np.multiply(m, cgs * a**aexp * h**hexp, dtype='f8')

    return m

