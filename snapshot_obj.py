import os
import re
from pathlib import Path
import glob
import numpy as np
import h5py

import dataset_compute

class Snapshot:

    def __init__(self, simID, snapID, name=""):
        """
        Parameters
        ----------
        simID : str
            Identifier of the simulation data -- should be equivalent to
            the directory name of the directory containing all the data 
            of the simulation.
        snapID : int
            The number identifying the snapshot.
        name : str, optional
            Label of the data set. If not given, is generated from simID
            and snapID.
        """

        self.simID = simID
        self.snapID = snapID
        # If not given, construct name from IDs:
        if not name:
            self.name = str(simID) + "_" + str(snapID)
        else:
            self.name = name

        # Initialize HDF5 files:
        self.grp_file = '.groups_{}_{}.hdf5'.format(simID,snapID)
        self.part_file = '.particles_{}_{}.hdf5'.format(simID,snapID)
        self.make_group_file()
        self.make_part_file()

    def make_group_file(self):
        """ Create a combined group data file and add links to all the
        actual data files """

        path = self.get_data_path('group')

        # Get files containing group data:
        files = \
                np.array(glob.glob(os.path.join(path,'eagle_subfind_tab*')))

        # Sort in ascending order:
        fnum = [int(fname.split(".")[-2]) for fname in files]
        sorting = np.argsort(fnum)
        files = files[sorting]

        # Create the file object with links to all the files:
        with h5py.File(self.grp_file,'a') as grpf:

            # Iterate through group files:
            for i,filename in enumerate(files):
                # Make an external link:
                if not 'link{}'.format(i) in grpf:
                    grpf['link{}'.format(i)] = h5py.ExternalLink(filename,'/')
        
    def make_part_file(self):
        """ Create a combined particle data file and add links to all 
        the actual data files """

        path = self.get_data_path('part')

        # Get files containing group data:
        files = \
                np.array(glob.glob(os.path.join(path,'snap_*.hdf5')))

        # Sort in ascending order:
        fnum = [int(fname.split(".")[-2]) for fname in files]
        sorting = np.argsort(fnum)
        files = files[sorting]

        # Create the file object with links to all the files:
        with h5py.File(self.part_file,'a') as partf:
    
            # Iterate through data files (and dismiss irrelevant files in the
            # directory):
            for i,filename in enumerate(files):
                # Make an external link:
                if not 'link{}'.format(i) in partf:
                    partf['link{}'.format(i)] = \
                            h5py.ExternalLink(filename,'/')

    def get_data_path(self, datatype):
        """ Constructs the path to data directory. 
        
        Paramaters
        ----------
        datatype : str
            recognized values are: 'part' and 'group'

        Returns
        -------
        path : str
            path to data directory
        """

        home = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(home,"snapshots",self.simID)

        prefix = ""
        if datatype == "part":
            prefix = "snapshot_"
        else:
            prefix += "groups_"

        # Find the snapshot directory and add to path:
        for dirname in os.listdir(path):
            if prefix + str(self.snapID) in dirname:
                path = os.path.join(path,dirname) 

        return path

    def file_of_halo(self, gn, sgn):
        """ Returns the file number of the file, which contains the halo
        identified by gn and sgn.

        Parameters
        ----------
        gn : int
            Halo group number
        sgn : int
            Halo subgroup number

        Returns
        -------
        fileNum : int
            File number
        """

        fileNum = -1

        with h5py.File(self.grp_file,'r') as grpf:

            links = [item for item in grpf.items() \
                    if ('link' in item[0])]

            for (name,link) in links:
                GNs = link['Subhalo/GroupNumber'][...]
                SGNs = link['Subhalo/SubGroupNumber'][...]

                if np.logical_and((GNs==gn),(SGNs==sgn)).sum() > 0:
                    fileNum = int(name.replace('link',''))
                    break

        return fileNum

    def get_subhalos(self, dataset, fnums=[]):
        """ Retrieves a dataset for subhaloes in the snapshot.
        
        Parameters
        ----------
        dataset : str
            Name of dataset to be retrieved.
        fnums : list of ints, optional
            Specifies files, which are to be read.

        Returns
        -------
        out : HDF5 datasets
            The requested dataset.
        """

        # Output array.
        out = np.empty((0))

        is_extension = False
        with h5py.File(self.grp_file,'r') as grpf:
            if dataset not in grpf['link1/Subhalo']:
                is_extension = True

        if is_extension:
            out = self.get_subhalo_extended(dataset)

        else:
            out = self.get_subhalo_catalogue(dataset,fnums)
            
        return out

    def get_subhalo_catalogue(self,dataset,fnums):
        """ Retrieves a dataset from the subhalo catalogues.

        Paramaters
        ----------
        dataset : str
            Name of dataset to be retrieved.
        fnums : list of ints, optional
            Specifies files, which are to be read.

        Returns
        -------
        out : HDF5 dataset
            Requested dataset in cgs units.
        """

        out = []
        link_names, link_sort = self.link_select(fnums)
    
        with h5py.File(self.grp_file,'r') as grpf:

            links = [f for (name,f) in grpf.items() \
                    if name in link_names]
            for f in links:
                tmp = f['Subhalo/{}'.format(dataset)][...]
                out.append(tmp)

        # Sort by link number:
        out = [out[i] for i in link_sort]
        
        # Combine to a single array.
        if len(out[0].shape) > 1:
            out = np.vstack(out)
        else:
            out = np.concatenate(out)

        out = self.convert_to_cgs_group(out,dataset)

        return out

    def get_subhalo_extended(self,dataset):
        """ Retrieves dataset from file if it is already calculated. 
        If not, calculates it and then returns it.

        Paramaters
        ----------
        dataset : str
            Name of (extended) dataset to be retrieved.

        Returns
        -------
        out : HDF5 dataset
            Requested dataset in cgs units.
        """

        out = []

        # Check if the dataset is already stored in grpf:
        in_grpf = False
        with h5py.File(self.grp_file,'r') as grpf:
            if 'extended/{}'.format(dataset) in grpf:
                out = grpf['extended/{}'.format(dataset)][...]
                in_grpf = True

        if not in_grpf:

            out = dataset_compute.generate_dataset(self,dataset)

            # Create dataset in grpf:
            with h5py.File(self.grp_file,'r+') as grpf:
                grpf.create_dataset('/extended/{}'.format(dataset),\
                        data=out)

        return out

    def link_select(self, fnums):
        """ Selects links from file keys and constructs an index list
        for sorting. """

        with h5py.File(self.grp_file,'r') as grpf:

            # Set to ndarray:
            keys = np.array(list(grpf.keys()))

            mask = [False for k in keys]
            for (idx,key) in enumerate(keys):
                if 'link' in key:
                    if len(fnums) == 0:
                        mask[idx] = True
                    elif int(key.replace('link','')) in fnums:
                        mask[idx] = True

        keys = keys[mask]
        linknums = [int(key.replace('link','')) for key in keys]
        sorting = np.argsort(linknums)

        return keys[sorting],sorting

    def convert_to_cgs_group(self,data,dataset):
        """ Read conversion factors for a halo dataset and convert it 
        into cgs units.

        Paramaters
        ----------
        data : HDF5 dataset
            Dataset to be converted.
        dataset : str
            Name of dataset.

        Returns
        -------
        converted : HDF5 dataset
            Dataset in cgs units.
        """

        converted = data

        with h5py.File(self.grp_file,'r') as grpf:

            # Get conversion factors.
            cgs     = grpf['link1/Subhalo/{}'.format(dataset)].attrs\
                    .get('CGSConversionFactor')
            aexp    = grpf['link1/Subhalo/{}'.format(dataset)].attrs\
                    .get('aexp-scale-exponent')
            hexp    = grpf['link1/Subhalo/{}'.format(dataset)].attrs\
                    .get('h-scale-exponent')
        
            # Get expansion factor and Hubble parameter from the 
            # header.
            a       = grpf['link1/Header'].attrs.get('Time')
            h       = grpf['link1/Header'].attrs.get('HubbleParam')
    
            # Convert to physical and return in cgs units.
            if data.dtype != np.int32 and data.dtype != np.int64:
                converted = np.multiply(data, cgs * a**aexp * h**hexp,\
                        dtype='f8')

        return converted

    def get_subhalos_IDs(self, fnums=[], part_type=[]):
        """ Read IDs of bound particles for each halo.
        
        Paramaters
        ----------
        fnums : list of ints, optional
            Specifies files, which are to be read
        part_type : list of int, optional
            Types of particles, whose attribute values are retrieved (the
            default is set for high-res part types)

        Returns
        -------
        IDs : HDF5 dataset 
            Dataset of lists of bound particles
        """

        IDs = []

        link_names_all, link_sort_all = self.link_select([])
        link_names_sel, link_sort_sel = self.link_select(fnums)
            
        with h5py.File(self.grp_file,'r') as grpf:

            # Get particle IDs from all files:
            particleIDs = []
            links = [f for (name,f) in grpf.items() \
                    if name in link_names_all]

            for link in links:
                particleIDs.append(link['IDs/ParticleID'][...])

            # Sort by link number:
            particleIDs = [particleIDs[i] for i in link_sort_all]

            particleIDs = np.concatenate(particleIDs)

            # Get IDs by halo from selected files:
            IDs = []
            links = [f for (name,f) in grpf.items() \
                    if name in link_names_sel]
            for i,link in enumerate(links):
                offset = link['Subhalo/SubOffset'][...]
                partNums = link['Subhalo/SubLength'][...]

                linkIDs = []
                # Select particles of given type:
                if part_type:
                    partNumsByType = link['Subhalo/SubLengthType'][...]

                    # Construct 2D array of offsets by part type for each
                    # subhalo:
                    pts = np.size(partNumsByType, axis=1)
                    cumsum = np.zeros((offset.size,pts+1)).astype(int)
                    cumsum[:,1:] = np.cumsum(partNumsByType,axis=1)
                    offsetByType = offset[:,np.newaxis] + cumsum

                    # For each subhalo, construct a list of indeces of the
                    # wanted particles:
                    construct_idx = lambda offs : \
                            [idx for i in part_type \
                            for idx in range(offs[i],offs[i+1])]

                    linkIDs = [particleIDs[construct_idx(o)] for o in \
                            offsetByType]

                else:
                    linkIDs = [particleIDs[o:o+n] for o,n in \
                            zip(offset,partNums)]
    
                IDs.append(linkIDs)
            
            # Sort by link number:
            IDs = [IDs[i] for i in link_sort_sel]
            IDs = np.concatenate(IDs)

        return IDs

    def get_particles(self, dataset, part_type=[0,1,4,5]):
        """ Reads the dataset from particle catalogues.
        
        Parameters
        ----------
        dataset : str
            Dataset to be retrieved.
        part_type : list of int, optional
            Types of particles, whose attribute values are retrieved (the
            default is set for high-res part types)

        Returns
        -------
        out : HDF5 dataset
            Requested dataset in cgs units.
        """

        # Output array.
        out = []

        # Get particle file:
        with h5py.File(self.part_file,'r') as partf:

            # For each particle type, loop over files, s.t. elements in
            # out are primarily ordered by particle type:
            for pt in part_type:
    
                for f in partf.values():
                    if 'PartType{}/{}'.format(pt,dataset) in f.keys():
                        tmp = f['PartType{}/{}'.format(pt,dataset)][...]
                        out.append(tmp)

        # Combine to a single array.
        if len(out[0].shape) > 1:
            out = np.vstack(out)
        else:
            out = np.concatenate(out)
            
        out = self.convert_to_cgs_part(out,dataset)
            
        return out

    def get_particle_masses(self,part_type=[0,1,4,5]):
        """ Reads particle masses in grams.
        
        Returns
        -------
        mass : HDF5 dataset
            Masses of each particle in cgs.
        """

        mass = []

        for pt in part_type:
            if pt in [1,2,3]:
                # Get dm particle masses:
                with h5py.File(self.part_file,'r') as partf:
                    dm_mass = partf['link1/Header']\
                            .attrs.get('MassTable')[pt]
                    dm_mass = self.convert_to_cgs_part(\
                            np.array([dm_mass]),'Masses')[0]
                    dm_n = partf['link1/Header']\
                            .attrs.get('NumPart_Total')[pt]
                    mass.append(np.ones(dm_n, dtype='f8')*dm_mass)
            else:
                mass.append(self.get_particles('Masses',part_type=[pt]))

        mass = np.concatenate(mass)

        return mass

    def convert_to_cgs_part(self,data,dataset):
        """ Read conversion factors for a dataset of particles and 
        convert it into cgs units.

        Paramaters
        ----------
        data : HDF5 dataset
            Dataset to be converted.
        dataset : str
            Name of dataset.

        Returns
        -------
        converted : HDF5 dataset
            Dataset in cgs units.
        """


        converted = data

        with h5py.File(self.part_file,'r') as partf:

            # Get conversion factors (same for all types):
            cgs     = partf['link1/PartType0/{}'.format(dataset)]\
                    .attrs.get('CGSConversionFactor')
            aexp    = partf['link1/PartType0/{}'.format(dataset)]\
                    .attrs.get('aexp-scale-exponent')
            hexp    = partf['link1/PartType0/{}'.format(dataset)]\
                    .attrs.get('h-scale-exponent')
            
            # Get expansion factor and Hubble parameter from the header:
            a       = partf['link1/Header'].attrs.get('Time')
            h       = partf['link1/Header'].attrs.get('HubbleParam')
        
            # Convert to physical and return in cgs units.
            if data.dtype != np.int32 and data.dtype != np.int64:
                converted = np.multiply(data, cgs * a**aexp * h**hexp,\
                        dtype='f8')

        return converted

