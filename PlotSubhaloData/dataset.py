import os
import re
from pathlib import Path
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

        home = os.path.dirname(os.getcwd())
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

    def get_subhalos(attr, divided):
        """ Retrieves the given attribute values for subhaloes in the
        dataset.
        
        Parameters
        ----------
        attr : str(, list)
            attribute(s) to be retrieved
        divided : bool
            if True, output is divided into satellites and isolated
            galaxies
        """

       # Use HDF5 external links to combine all files into a single File
       # object. Make sure that no additional hierarchy between the files
       # is introduced. Also, note that the files have groups with the
       # same names that we would actually want to identify. How can we
       # accomplish this?


