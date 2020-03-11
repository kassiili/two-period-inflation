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

        # Define class attributes:
        self.simID = simID
        self.snapID = snapID
        if not name:
            self.name = simID + "_" + snapID
        else:
            self.name = name
        self.grp_file = 'groups_{}.hdf5'.format(name)
        self.part_file = 'particles_{}.hdf5'.format(name)

        # Initialize HDF5 files:
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
                if (gn in link['Subhalo/GroupNumber']) and \
                        (sgn in link['Subhalo/SubGroupNumber']):
                    fileNum = int(name[-1])
                    break

        return fileNum

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
        path = os.path.join(home,"snapshots",self.simID)

        prefix = ""
        if datatype == "part":
            prefix = "snapshot_"
        elif datatype == "group":
            prefix += "groups_"
        else:
            return None

        # Find the snapshot directory and add to path:
        for dirname in os.listdir(path):
            if prefix + str(self.snapID) in dirname:
                path = os.path.join(path,dirname) 

        return path

    def get_subhalos(self, attr, divided=True, fnums=[]):
        """ Retrieves the given attribute values for subhaloes in the
        dataset.
        
        Parameters
        ----------
        attr : str
            attribute to be retrieved
        divided : bool, optional
            if True (default), output is divided into satellites and 
            isolated galaxies
        fnums : list of ints, optional
            Specifies files, which are to be read

        Returns
        -------
        data : tuple of HDF5 datasets
            tuple of one or two entries, depending on the value of the
            argument "divided". If divided == True, return satellite data
            in the first entry and isolated galaxies data in the second.
        """

        data = self.read_subhalo_attr(attr, fnums=fnums)

        if divided:
            SGNs = self.read_subhalo_attr('SubGroupNumber')

            # Divide into satellites and isolated galaxies:
            dataSat = data[SGNs != 0]
            dataIsol = data[SGNs == 0]

            return (dataSat,dataIsol)
        else:
            return (data,)

    def read_subhalo_attr(self, attr, fnums=[]):
        """ Reads the data files for the attribute "attr" of each subhalo.
        
        Parameters
        ----------
        attr : str
            attribute to be retrieved
        fnums : list of ints, optional
            Specifies files, which are to be read

        Returns
        -------
        out : HDF5 dataset
            dataset of the values of attribute "attr" for each subhalo
        """

        # Output array.
        out = []

        is_extension = False
        with h5py.File(self.grp_file,'r') as grpf:
            if attr not in grpf['link1/Subhalo']:
                is_extension = True

        if is_extension:

           out = self.read_subhalo_extended_attr(attr)

        else:
    
            with h5py.File(self.grp_file,'r') as grpf:

                names = [name for name in grpf.keys() \
                        if ('link' in name)]

                # Look only in the requested files:
                if len(fnums) > 0:
                    names = [name for name in names if \
                            int(name.replace('link','')) in fnums]
                
                links = [f for (name,f) in grpf.items() if name in names]
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

    def read_subhalo_extended_attr(self,attr):
        """ Retrieves dataset corresponding to attr from file if it is
        already calculated. If not, calculates it and then returns it.

        Paramaters
        ----------
        datatype : attr
            (extended) attribute to be retrieved

        Returns
        -------
        out : HDF5 dataset
            dataset of the values of attribute "attr" for each subhalo
        """

        out = []

        # Check if velocities at 1kpc are already stored in grpf:
        attr_in_grpf = False
        with h5py.File(self.grp_file,'r') as grpf:
            if 'extended/{}'.format(attr) in grpf:
                out = grpf['extended/{}'.format(attr)][...]
                attr_in_grpf = True

        if not attr_in_grpf:

            out = dataset_compute.calculate_attr(self,attr)

            # Create dataset in grpf:
            with h5py.File(self.grp_file,'r+') as grpf:
                if '/extended' not in grpf:
                    ext = grpf.create_group('extended')
                    attr_dataset = \
                            ext.create_dataset(attr, data=out)
                else:
                    attr_dataset = grpf.create_dataset(\
                            '/extended/{}'.format(attr), data=out)

        return out

    def get_subhalos_IDs(self, fnums=[]):

        IDs = []
            
        with h5py.File(self.grp_file,'r') as grpf:

            # Get files that are in order before fnum:
            names = [name for name in grpf.keys() \
                    if ('link' in name)]
            links = [f for (name,f) in grpf.items() if name in names]

            # Get particle IDs:
            particleIDs = []
            for link in links:
                particleIDs.append(link['IDs/ParticleID'][...])

            particleIDs = np.concatenate(particleIDs)

            # Look only in the requested files:
            if len(fnums) > 0:
                names = [name for name in names if \
                        int(name.replace('link','')) in fnums]
            links = [f for (name,f) in grpf.items() if name in names]

            IDs = []
            for i,link in enumerate(links):
                offset = link['Subhalo/SubOffset'][...]
                partNums = link['Subhalo/SubLength'][...]

                splitByStart = np.split(particleIDs,offset)[1:]
                splitByEnd = np.split(particleIDs,offset+partNums)[:-1]
                linkIDs = [np.intersect1d(bystart,byend) for (bystart,byend)\
                        in zip(splitByStart,splitByEnd)]
    
                IDs.append(linkIDs)
            
            IDs = np.concatenate(IDs)

        return IDs

    def get_particles(self, attr, part_type=[0,1,4,5]):
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
    
            # Loop over each file and extract the data.
            for f in partf.values():
                
                # Loop over particle types:
                for pt in part_type:
                    if attr in f['PartType{}'.format(pt)].keys():
                        tmp = f['PartType{}/{}'.format(pt,attr)][...]
                        out.append(tmp)

        # Combine to a single array.
        if len(out[0].shape) > 1:
            out = np.vstack(out)
        else:
            out = np.concatenate(out)
            
        out = self.convert_to_cgs_part(out,attr)
            
        return out

    def convert_to_cgs_part(self,data,attr):

        converted = data

        with h5py.File(self.part_file,'r') as partf:

            # Get conversion factors (same for all types):
            cgs     = partf['link1/PartType0/{}'.format(attr)]\
                    .attrs.get('CGSConversionFactor')
            aexp    = partf['link1/PartType0/{}'.format(attr)]\
                    .attrs.get('aexp-scale-exponent')
            hexp    = partf['link1/PartType0/{}'.format(attr)]\
                    .attrs.get('h-scale-exponent')
            
            # Get expansion factor and Hubble parameter from the header:
            a       = partf['link1/Header'].attrs.get('Time')
            h       = partf['link1/Header'].attrs.get('HubbleParam')
        
            # Convert to physical and return in cgs units.
            if data.dtype != np.int32 and data.dtype != np.int64:
                converted = np.multiply(data, cgs * a**aexp * h**hexp, dtype='f8')

        return converted

    def get_particle_masses(self,part_type=[0,1,4,5]):
        """ Reads particle masses, ignoring types 2,3!!
        
        Returns
        -------
        mass : HDF5 dataset
            masses of each particle
        """

        mass = []

        for pt in part_type:
            if pt in [1,2,3]:
                # Get dm particle masses:
                with h5py.File(self.part_file,'r') as partf:
                    dm_mass = partf['link1/Header']\
                            .attrs.get('MassTable')[1]
                    dm_n = partf['link1/Header']\
                            .attrs.get('NumPart_Total')[1]
                    mass = np.concatenate((mass,\
                                np.ones(dm_n, dtype='f8')*dm_mass))
            else:
                mass = np.concatenate((mass,\
                        self.get_particles('Masses',part_type=[4])))

        mass = self.convert_to_cgs_part(mass,'Masses')

        return mass

