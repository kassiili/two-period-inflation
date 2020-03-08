import os
import re
from pathlib import Path
import glob
import numpy as np
import h5py

import astropy.units as u
from astropy.constants import G

class Dataset:

    def __init__(self, ID, name):
        """
        Parameters
        ----------
        ID : str
            identifier of the simulation data -- should be equivalent to
            the directory name of the snapshot
        name : str
            label of the data set
        """

        self.ID = ID
        self.name = name
        self.grp_file = 'groups_{}.hdf5'.format(name)
        self.part_file = 'particles_{}.hdf5'.format(name)
        self.make_group_file()
        self.make_part_file()

    def make_part_file(self):
        """ Create a combined particle data file and add links to all 
        the actual data files """

        # Create the file object with links to all the files:
        with h5py.File(self.part_file,'a') as partf:
            path = self.get_data_path('part')
    
            # Iterate through data files (and dismiss irrelevant files in the
            # directory):
            for i,filename in \
            enumerate(glob.glob(os.path.join(path,'snap_*.hdf5'))):
                # Make an external link:
                if not 'link{}'.format(i) in partf:
                    partf['link{}'.format(i)] = \
                            h5py.ExternalLink(filename,'/')

    def make_group_file(self):
        """ Create a combined group data file and add links to all the
        actual data files """

        # Create the file object with links to all the files:
        with h5py.File(self.grp_file,'a') as grpf:
            path = self.get_data_path('group')
    
            # Iterate through group files (and dismiss irrelevant files in the
            # directory):
            for i,filename in \
            enumerate(glob.glob(os.path.join(path,'eagle_subfind_tab*'))):
                # Make an external link:
                if not 'link{}'.format(i) in grpf:
                    grpf['link{}'.format(i)] = h5py.ExternalLink(filename,'/')
        
    def count_files(self):
        """ Count the relevant data files in the head directory of the
        dataset. 

        Returns
        -------
        cnt : int
            number of data files
        """

        groupfs = os.listdir(self.get_data_path("group"))
        partfs = os.listdir(self.get_data_path("part"))

        cnt = {}
        # Exclude irrelevant files and count:
        cnt["group"] = \
        sum([bool(re.match("eagle_subfind_tab_127_z000p000.*",f)) \
            for f in groupfs])

        cnt["part"] = sum([bool(re.match("snap_127_z000p000.*",f)) \
            for f in partfs])

        return cnt

    def get_data_path(self, datatype):
        """ Constructs the path to data directory. 
        
        Paramaters
        ----------
        datatype : str
            recognized values are: part and group

        Returns
        -------
        path : str
            path to data directory
        """

        home = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(home,"snapshots",self.ID)
        direc = ""
        if datatype == "part":
            direc = "snapshot_"
        elif datatype == "group":
            direc += "groups_"
        else:
            return None

        fields = self.ID.split("_")
        direc += fields[-2] + "_" + fields[-1]

        return os.path.join(path,direc)

    def get_file_prefix(self, datatype):
        """ Constructs file prefix for individual data files. 
        
        Paramaters
        ----------
        datatype : str
            recognized values are: part and group

        Returns
        -------
        prefix : str
            file prefix
        """

        prefix = ""
        if datatype == "part":
            prefix = "snap"
        elif datatype == "group":
            prefix += "eagle_subfind_tab_"
        else:
            return None

        fields = self.ID.split("_")
        prefix += fields[-2] + "_" + fields[-1]

        return prefix

    def get_subhalos(self, attr, divided=True):
        """ Retrieves the given attribute values for subhaloes in the
        dataset.
        
        Parameters
        ----------
        attr : str
            attribute to be retrieved
        divided : bool, optional
            if True (default), output is divided into satellites and 
            isolated galaxies

        Returns
        -------
        data : tuple of HDF5 datasets
            tuple of one or two entries, depending on the value of the
            argument "divided". If divided == True, return satellite data
            in the first entry and isolated galaxies data in the second.
        """

        data = self.read_subhalo_attr(attr)

        if divided:
            SGNs = self.read_subhalo_attr('SubGroupNumber')

            # Divide into satellites and isolated galaxies:
            dataSat = data[SGNs != 0]
            dataIsol = data[SGNs == 0]

            return (dataSat,dataIsol)
        else:
            return (data,)

    def read_subhalo_attr(self, attr):
        """ Reads the data files for the attribute "attr" of each subhalo.
        
        Parameters
        ----------
        attr : str
            attribute to be retrieved

        Returns
        -------
        out : HDF5 dataset
            dataset of the values of attribute "attr" for each subhalo
        """

        # Output array.
        out = []

        if attr == 'V1kpc':

           out = self.get_subhalo_v1kpc()

        else:
    
            # Get group file:
            with h5py.File(self.grp_file,'r') as grpf:

                # Loop over each file and extract the data.
                links = (f for f in grpf.values() if ('Subhalo' in f))
                for f in links:
                    tmp = f['Subhalo/{}'.format(attr)][...]
                    out.append(tmp)
            
                # Get conversion factors.
                cgs     = grpf['link1/Subhalo/{}'.format(attr)].attrs\
                        .get('CGSConversionFactor')
                aexp    = grpf['link1/Subhalo/{}'.format(attr)].attrs\
                        .get('aexp-scale-exponent')
                hexp    = grpf['link1/Subhalo/{}'.format(attr)].attrs\
                        .get('h-scale-exponent')
            
                # Get expansion factor and Hubble parameter from the 
                # header.
                a       = grpf['link1/Header'].attrs.get('Time')
                h       = grpf['link1/Header'].attrs.get('HubbleParam')
        
                # Combine to a single array.
                if len(out[0].shape) > 1:
                    out = np.vstack(out)
                else:
                    out = np.concatenate(out)
        
                # Convert to physical and return in cgs units.
                if out.dtype != np.int32 and out.dtype != np.int64:
                    out = np.multiply(out, cgs * a**aexp * h**hexp, dtype='f8')

        return out

    def get_subhalo_v1kpc(self):
        """ Retrieves V1kpc dataset from file if it is already calculated,
        if not, calculates it and then returns it. """

        v1kpc = []

        # Check if velocities at 1kpc are already stored in grpf:
        v1kpc_in_grpf = False
        with h5py.File(self.grp_file,'r') as grpf:
            if 'extended/V1kpc' in grpf:
                v1kpc = grpf['extended/V1kpc'][...]
                v1kpc_in_grpf = True

        if not v1kpc_in_grpf:

            v1kpc = self.calcVelocitiesAt1kpc()

            # Create V1kpc dataset in grpf:
            with h5py.File(self.grp_file,'r+') as grpf:
                if '/extended' not in grpf:
                    ext = grpf.create_group('extended')
                    v1kpc_dataset = \
                            ext.create_dataset('V1kpc', data=v1kpc)
                else:
                    v1kpc_dataset = grpf.create_dataset(\
                            '/extended/V1kpc', data=v1kpc)

        return v1kpc

    def calcVelocitiesAt1kpc_fake(self):
        return np.arange(1000)

    def get_particles(self, attr, part_type=[0,1,2,3,4,5]):
        """ Reads the data files for the attribute "attr" of each particle.
        
        Parameters
        ----------
        attr : str
            attribute to be retrieved
        part_type : list of int, optional
            types of particles, whose attribute values are retrieved (the
            default is all of them)

        Returns
        -------
        out : HDF5 dataset
            dataset of the values of attribute "attr" for each particle 
        """

        # Output array.
        out = []

        # Get particle file:
        with h5py.File(self.part_file,'r') as partf:

            out = self.get_particle_attributes(partf,attr,part_type)
            
            # Get conversion factors.
            cgs     = partf['link1/PartType{}/{}'.format(\
                    part_type[0],attr)].attrs.get('CGSConversionFactor')
            aexp    = partf['link1/PartType{}/{}'.format(\
                    part_type[0],attr)].attrs.get('aexp-scale-exponent')
            hexp    = partf['link1/PartType{}/{}'.format(\
                    part_type[0],attr)].attrs.get('h-scale-exponent')
            
            # Get expansion factor and Hubble parameter from the 
            # header.
            a       = partf['link1/Header'].attrs.get('Time')
            h       = partf['link1/Header'].attrs.get('HubbleParam')
        
            # Combine to a single array.
            if len(out[0].shape) > 1:
                out = np.vstack(out)
            else:
                out = np.concatenate(out)
            
            # Convert to physical and return in cgs units.
            if out.dtype != np.int32 and out.dtype != np.int64:
                out = np.multiply(out, cgs * a**aexp * h**hexp, dtype='f8')
            
        return out

    def is_particle_attribute(self,files,attr):
        """ Check some of the particle types has attribute. 

        Parameters
        ----------
        attr : str
            attribute to be checked
        files : HDF5 File
            file object with links to all particle data files
        part_type : list of int
            types of particles, whose attribute values are retrieved 

        Returns
        -------
        is_part_attr : bool
            is True, if attr is a particle attribute
        """
        
        # Get all particle attributes:
        part_attrs = []
        for pt in [0,1,2,3,4,5]:
            part_attrs = part_attrs + \
                    list(files['link1/PartType{}'.format(pt)].keys())

        is_part_attr = attr in part_attrs
        return is_part_attr

    def get_particle_attributes(self,files,attr,part_type):
        """ A help function for retrieving particle attributes,
        specifically, from particle data files.

        Parameters
        ----------
        attr : str
            attribute to be retrieved
        files : HDF5 File
            file object with links to all particle data files

        Returns
        -------
        out : HDF5 dataset
            dataset of particle attribute values
        """

        out = []

        # Loop over each file and extract the data.
        for f in files.values():
            
            # Loop over particle types:
            for pt in part_type:
                if attr in f['PartType{}'.format(pt)].keys():
                    tmp = f['PartType{}/{}'.format(pt,attr)][...]
                    out.append(tmp)

        return out

    def get_particle_masses(self):
        # Get gas particle masses:
        mass = self.get_particles('Masses',part_type=[0])

        # Get dm particle masses:
        with h5py.File(self.part_file,'r') as partf:
            for i in range(1,4):
                dm_mass = partf['link1/Header'].attrs.get('MassTable')[i]
                dm_n = partf['link1/Header'].attrs.get('NumPart_Total')[i]
                mass = np.concatenate((mass,\
                        np.ones(dm_n, dtype='f8')*dm_mass))
        
        # Get star and BH particle masses:
        mass = np.concatenate((mass,\
                self.get_particles('Masses',part_type=[4])))
        mass = np.concatenate((mass,\
                self.get_particles('Masses',part_type=[5])))

        return mass

    def get_subhalo_part_idx(self):

        # Get subhalos:
        halo_gns = self.get_subhalos('GroupNumber',\
                divided=False)[0].astype(int)
        halo_sgns = self.get_subhalos('SubGroupNumber',\
                divided=False)[0].astype(int)

        # Get particle data:
        part_gns = self.get_particles('GroupNumber')
        part_sgns = self.get_particles('SubGroupNumber')

        # Get halo indices:
        sorting = np.lexsort((halo_sgns,halo_gns))

        # Invert sorting:
        inv_sorting = [0] * len(sorting)
        for idx, val in enumerate(sorting):
            inv_sorting[val] = idx

        # Loop through particles and save indices to lists. Halos in the
        # list behind the part_idx key are arranged in ascending order 
        # with gn and sgn, i.e. in the order lexsort would arrange them:
        gn_count = np.bincount(halo_gns)
        part_idx = [[] for i in range(halo_gns.size)]
        for idx, (gn,sgn) in enumerate(zip(part_gns,part_sgns)):
            # Exclude unbounded particles (for unbounded: sgn = max_int):
            if sgn < 10**6:
                part_idx[(gn-1)*gn_count[gn]+sgn].append(idx)

        # Convert to ndarray and sort in order corresponding to the halo
        # datasets:
        part_idx = np.array(part_idx)[inv_sorting]
        return part_idx

    def calcVelocitiesAt1kpc(self):
        """ For each subhalo, calculate the circular velocity at 1kpc. 
        Assume that there are no jumps in the SubGroupNumber values in any
        of the groups."""

        # Get particle data:
        part = {}
        part['gns'] = self.get_particles('GroupNumber')
        part['sgns'] = self.get_particles('SubGroupNumber')
        part['coords'] = self.get_particles('Coordinates') \
                * u.cm.to(u.kpc)
        part['mass'] = self.get_particle_masses() * u.g.to(u.Msun)

        halo = {}
        halo['COPs'] = self.get_subhalos('CentreOfPotential',\
                divided=False)[0] * u.cm.to(u.kpc)
        halo['part_idx'] = self.get_subhalo_part_idx()

        massWithin1kpc = np.zeros((halo['COPs'][:,0].size))

        for idx, (cop,idx_list) in enumerate(\
                zip(halo['COPs'],halo['part_idx'])):

            # Get coords and mass of the particles in the corresponding halo:
            coords = part['coords'][idx_list]
            mass = part['mass'][idx_list]

            # Calculate distances to COP:
            r = np.linalg.norm(coords - cop, axis=1)

            # Get coordinates within 1kpc from COP:
            r1kpc_mask = np.logical_and(r > 0, r < 1)

            massWithin1kpc[idx] = mass[r1kpc_mask].sum()

        myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
        v1kpc = np.sqrt(massWithin1kpc * myG)

        return v1kpc
        

    def calcVelocitiesAt1kpc_slow(self):
        """ For each subhalo, calculate the circular velocity at 1kpc. """

        # Get particle data:
        part = {}
        part['gns'] = self.get_particles('GroupNumber')
        part['sgns'] = self.get_particles('SubGroupNumber')
        part['coords'] = self.get_particles('Coordinates')

        # Get gas particle masses:
        part['mass'] = self.get_particles('Masses',part_type=[0])

        # Get dm particle masses:
        with h5py.File(self.part_file,'r') as partf:
            for i in range(1,4):
                dm_mass = partf['link1/Header'].attrs.get('MassTable')[i]
                dm_n = partf['link1/Header'].attrs.get('NumPart_Total')[i]
                part['mass'] = np.concatenate((part['mass'],\
                        np.ones(dm_n, dtype='f8')*dm_mass))
        
        # Get star and BH particle masses:
        part['mass'] = np.concatenate((part['mass'],\
                self.get_particles('Masses',part_type=[4])))
        part['mass'] = np.concatenate((part['mass'],\
                self.get_particles('Masses',part_type=[5])))

        # Get subhalodata:
        halo = {}
        halo['gns'] = self.get_subhalos('GroupNumber',divided=False)[0]
        halo['sgns'] = \
                self.get_subhalos('SubGroupNumber',divided=False)[0]
        halo['COPs'] = \
                self.get_subhalos('CentreOfPotential',divided=False)[0]

        massWithin1kpc = np.zeros((halo['gns'].size))

        # Loop through subhalos:
        for idx, (gn, sgn, cop) in \
                enumerate(zip(halo['gns'],halo['sgns'],halo['COPs'])):

            # Get coordinates and masses of the particles in the halo:
            halo_mask = np.logical_and(part['gns'] == gn, \
                    part['sgns'] == sgn)
            coords = part['coords'][halo_mask]
            mass = part['mass'][halo_mask]

            # Calculate distances to COP:
            r = np.linalg.norm(coords - cop, axis=1)

            # Get coordinates within 1kpc from COP:
            r1kpc_mask = np.logical_and(r > 0, r < 1)

            massWithin1kpc[idx] = mass[r1kpc_mask].sum()

        myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
        v1kpc = np.sqrt(massWithin1kpc * myG)

        return v1kpc
        
