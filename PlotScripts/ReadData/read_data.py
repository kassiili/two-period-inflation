import numpy as np
import h5py
import os

# Numbers of files for different datasets:
#   - Particle data:
#       * V1_LR_fix and V1_MR_fix: 16
#   - Group data:
#       * V1_LR_fix: 96
#       * V1_MR_fix: 192

class read_data:

    # The path (from the project's "home directory" practise-with-datasets) to the dataset is given as an argument:
    def __init__(self, dataset='snapshots/V1_LR_fix_127_z000p000', nfiles=16):

        self.nfiles=nfiles
        fields = dataset.split("_")

        # Set relative paths to data files:
        self.part_data_path = dataset + "/" + "snapshot_" + fields[-2] + "_" + fields[-1]
        self.group_data_path = dataset + "/" + "groups_" + fields[-2] + "_" + fields[-1]

        # Set file prefixes:
        self.part_file_prefix = "snap_" + fields[-2] + "_" + fields[-1]
        self.group_file_prefix = "eagle_subfind_tab_" + fields[-2] + "_" + fields[-1]
    
    def read_header(self):
        """ Read various attributes from the header group. """
    
        path = self.get_path(self.part_data_path)
    
        filename = '%s/%s.0.hdf5'%(path, self.part_file_prefix)
        f = h5py.File(filename, 'r')
        a       = f['Header'].attrs.get('Time')         # Scale factor.
        h       = f['Header'].attrs.get('HubbleParam')  # h.
        mass    = f['Header'].attrs.get('MassTable')    # 10^10 Msun
        boxsize = f['Header'].attrs.get('BoxSize')      # L [Mph/h].
        f.close()
    
        return a, h, mass, boxsize

    def read_dataset(self, itype, att):
        """ Read particle data, itype is the PartType and att is the attribute name. """
    
        # Output array.
        data = []
    
        path = self.get_path(self.part_data_path)
    
        # Loop over each file and extract the data.
        for i in range(self.nfiles):
            filename = '%s/%s.%i.hdf5'%(path, self.part_file_prefix, i)
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
    
    def read_dataset_dm_mass(self):
        """ Special case for the mass of dark matter particles. """
    
        path = self.get_path(self.part_data_path)
    
        filename = '%s/%s.0.hdf5'%(path, self.part_file_prefix)
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
    
    def read_subhaloData(self, att):
        """ Read a selected dataset from the subhalo catalogues, att is the attribute name, dataset determines, which dataset is used.  """
    
        path = self.get_path(self.group_data_path)
    
        # Output array.
        data = []
    
        # Loop over each file and extract the data.
        for i in range(self.nfiles):
            filename = '%s/%s.%i.hdf5'%(path, self.group_file_prefix, i)
            f = h5py.File(filename, 'r')
            tmp = f['Subhalo/%s'%att][...]
            data.append(tmp)
    
            # Get conversion factors.
            cgs     = f['Subhalo/%s'%att].attrs.get('CGSConversionFactor')
            aexp    = f['Subhalo/%s'%att].attrs.get('aexp-scale-exponent')
            hexp    = f['Subhalo/%s'%att].attrs.get('h-scale-exponent')
    
            # Get expansion factor and Hubble parameter from the header.
            a       = f['Header'].attrs.get('Time')
            h       = f['Header'].attrs.get('HubbleParam')
    
            f.close()
    
        # Combine to a single array.
        if len(tmp.shape) > 1:
            data = np.vstack(data)
        else:
            data = np.concatenate(data)
    
        # Convert to physical and return in cgs units.
        if data.dtype != np.int32 and data.dtype != np.int64:
            data = np.multiply(data, cgs * a**aexp * h**hexp, dtype='f8')
    
        return data
    
    def read_partIDs(self, att):
        """ Read a selected dataset from the IDs group of the group data files, att is the attribute name.  """
    
        path = self.get_path(self.group_data_path)
    
        # Output array.
        data = []
    
        # Loop over each file and extract the data.
        for i in range(self.nfiles):
            filename = '%s/%s.%i.hdf5'%(path, self.group_file_prefix, i)
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

    def get_path(self, path):
        """ Add relative path from current directory to the directory where the data is located. """
        dirname = os.path.dirname(__file__)
        if not dirname:
            new_path = '../../%s'%path 
        else:
            new_path = '%s/../../%s'%(dirname,path)
    
        return new_path
    
#reader = Read_data()
#print(reader.read_dataset_dm_mass())
