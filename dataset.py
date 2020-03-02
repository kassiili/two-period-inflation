import os
import re
from pathlib import Path
import glob
import numpy as np
import h5py

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
        self.make_group_file()
        
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

    def get_subhalos(self, attr, divided):
        """ Retrieves the given attribute values for subhaloes in the
        dataset.
        
        Parameters
        ----------
        attr : str
            attribute to be retrieved
        divided : bool
            if True, output is divided into satellites and isolated
            galaxies

        Returns
        -------
        data : tuple
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

        # Get group file:
        with h5py.File(self.grp_file,'r') as grpf:

            # Loop over each file and extract the data.
            for f in grpf.values():
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

    def make_group_file(self):
        """ Create a group file and add links to all the data files """

        # Create the file object with links to all the files:
        grpf = h5py.File(self.grp_file,'w')
        path = self.get_data_path('group')

        # Iterate through group files (and dismiss irrelevant files in the
        # directory):
        for i,filename in \
        enumerate(glob.glob(os.path.join(path,'eagle_subfind_tab*'))):
            # Make an external link:
            grpf['link{}'.format(i)] = h5py.ExternalLink(filename,'/')

        grpf.close()

