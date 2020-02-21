import numpy as np
import h5py
import os

def read_dataset_dm_mass(dataset='V1_LR_fix/snapshot_127_z000p000'):
    """ Special case for the mass of dark matter particles. """

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

#data = read_dataset_dm_mass()
#print(data[0])
