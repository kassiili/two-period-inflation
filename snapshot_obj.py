import os
import re
from pathlib import Path
import glob
import numpy as np
import h5py

import dataset_compute
import data_file_manipulation


class Snapshot:
    """ Object that represents the data of a single snapshot. 

    Attributes
    ----------
    sim_id : str
        Identifier of the simulation data equivalent to the directory name
        of the directory containing all the data of the simulation.
    snap_id : int
        The number identifying the snapshot.
    name : str, optional
        Label of the data set.
    grp_file : str
        Filename of the combined group data file.
    part_file : hdf5 file
        Filename of the combined particle data file.

    """

    def __init__(self, sim_id, snap_id, name=""):
        """
        Parameters
        ----------
        sim_id : str
            Identifier of the simulation data -- should be equivalent to
            the directory name of the directory containing all the data 
            of the simulation.
        snap_id : int
            The number identifying the snapshot.
        name : str, optional
            Label of the data set. If not given, is generated from simID
            and snapID.
        """

        self.sim_id = sim_id
        self.snap_id = snap_id
        # If not given, construct name from IDs:
        if not name:
            self.name = str(sim_id) + "_" + str(snap_id)
        else:
            self.name = name

        # Generate combined data files:
        self.grp_file = '.groups_{}_{}.hdf5'.format(sim_id, snap_id)
        self.part_file = '.particles_{}_{}.hdf5'.format(sim_id, snap_id)

        path = data_file_manipulation.get_data_path('group', sim_id, snap_id)
        data_file_manipulation.combine_data_files( \
            np.array(glob.glob(os.path.join(path, 'eagle_subfind_tab*'))), \
            self.grp_file)

        path = data_file_manipulation.get_data_path('part', sim_id, snap_id)
        data_file_manipulation.combine_data_files( \
            np.array(glob.glob(os.path.join(path, 'snap*'))), \
            self.part_file)

    def get_subhalos(self, dataset, fnums=[], units='cgs'):
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
        with h5py.File(self.grp_file, 'r') as grpf:
            if dataset not in grpf['link1/Subhalo']:
                is_extension = True

        if is_extension:
            out = self.get_subhalo_extended(dataset)

        else:
            out = self.get_subhalo_catalogue(dataset, fnums, units)

        return out

    def get_subhalo_catalogue(self, dataset, fnums, units):
        """ Retrieves a dataset from the subhalo catalogues.

        Parameters
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
        link_names, link_sort = self.link_select('group', fnums)

        with h5py.File(self.grp_file, 'r') as grpf:

            links = [f for (name, f) in grpf.items() \
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

        if units == 'cgs':
            out = self.convert_to_cgs_group(out, dataset)

        return out

    def get_subhalo_extended(self, dataset):
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
        with h5py.File(self.grp_file, 'r') as grpf:
            if 'extended/{}'.format(dataset) in grpf:
                out = grpf['extended/{}'.format(dataset)][...]
                in_grpf = True

        if not in_grpf:
            out = dataset_compute.generate_dataset(self, dataset)

            # Create dataset in grpf:
            with h5py.File(self.grp_file, 'r+') as grpf:
                grpf.create_dataset('/extended/{}'.format(dataset), \
                                    data=out)

        return out

    def get_subhalos_IDs(self, part_type, fnums=[]):
        """ Read IDs of bound particles of given type for each halo.
        
        Paramaters
        ----------
        part_type : list of int
            Types of particles, whose attribute values are retrieved (the
            default is set for high-res part types)
        fnums : list of ints, optional
            Specifies files, which are to be read

        Returns
        -------
        IDs : HDF5 dataset 
            Dataset of ndarrays of bound particles
        """

        IDs_bound = self.get_bound_particles("ParticleID")

        # Construct mask for selecting bound particles of type pt:
        IDs_pt = self.get_particles("ParticleIDs", part_type=[part_type])
        mask_pt = np.isin(IDs_bound, IDs_pt)

        IDs = []
        link_names_sel, link_sort_sel = self.link_select('group', fnums)
        with h5py.File(self.grp_file, 'r') as grpf:
            # Get IDs by halo from selected files:
            links = [f for (name, f) in grpf.items() \
                     if name in link_names_sel]
            for i, link in enumerate(links):
                linkIDs = []

                offset = link['Subhalo/SubOffset'][...]
                pnum = link['Subhalo/SubLength'][...]

                select_halo = lambda o, n: \
                    IDs_bound[o:o + n][mask_pt[o:o + n]]
                linkIDs = [select_halo(o, n) for o, n in \
                           zip(offset, pnum)]

                IDs.append(linkIDs)

            # Sort by link number:
            IDs = [IDs[i] for i in link_sort_sel]
            IDs = [ids for link in IDs for ids in link]

        return np.array(IDs)

    def get_bound_particles(self, dataset):
        """ Reads data entries for bound particles in the group catalogues 

        Parameters
        ----------
        dataset : str
            Dataset to be retrieved.

        Returns
        -------
        out : HDF5 dataset
            Requested dataset in cgs units.
        """

        out = []
        link_names_all, link_sort_all = self.link_select('group', [])

        with h5py.File(self.grp_file, 'r') as grpf:
            # Get particle IDs from all files:
            out = []
            links = [f for (name, f) in grpf.items() \
                     if name in link_names_all]

            for link in links:
                out.append(link['IDs/{}'.format(dataset)][...])

        # Sort by link number:
        out = [out[i] for i in link_sort_all]
        out = np.concatenate(out)

        out = self.convert_to_cgs_bound(out, dataset)

        return out

    def get_halo_number(self, which_gns):

        n = 0
        gns = self.get_subhalos("GroupNumber")
        if not which_gns:
            n = gns.size
        else:
            n = np.sum(np.isin(gns, which_gns))

        return n

    def get_particles(self, dataset, part_type=[0, 1, 4, 5], fnums=[]):
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

        link_names, link_sort = self.link_select('part', fnums)

        # Get particle file:
        with h5py.File(self.part_file, 'r') as partf:

            # For each particle type, loop over files, s.t. elements in
            # out are primarily ordered by particle type:
            for pt in part_type:

                pt_out = []
                links = [f for (name, f) in partf.items() \
                         if name in link_names]
                for f in links:
                    if 'PartType{}/{}'.format(pt, dataset) in f.keys():
                        tmp = f['PartType{}/{}'.format(pt, dataset)][...]
                        pt_out.append(tmp)

                # Sort by link number and add to main list:
                out += [pt_out[i] for i in link_sort]

        # Combine to a single array.
        if len(out[0].shape) > 1:
            out = np.vstack(out)
        else:
            out = np.concatenate(out)

        out = self.convert_to_cgs_part(out, dataset)

        return out

    def get_particle_masses(self, part_type=[0, 1, 4, 5]):
        """ Reads particle masses in grams.
        
        Returns
        -------
        mass : HDF5 dataset
            Masses of each particle in cgs.
        """

        mass = []

        for pt in part_type:
            if pt in [1]:
                # Get dm particle masses:
                with h5py.File(self.part_file, 'r') as partf:
                    dm_mass = partf['link1/Header'] \
                        .attrs.get('MassTable')[pt]
                    dm_mass = self.convert_to_cgs_part(
                        np.array([dm_mass]), 'Masses')[0]
                    dm_n = partf['link1/Header'] \
                        .attrs.get('NumPart_Total')[pt]
                    mass.append(np.ones(dm_n, dtype='f8') * dm_mass)
            else:
                mass.append(self.get_particles('Masses', part_type=[pt]))

        mass = np.concatenate(mass)

        return mass

    def get_attribute(self, attr, entry, data_category='group'):
        """ Reads an attribute of the given entry (either dataset or
        group).

        Returns
        -------
        out : float
            The value of the attribute.
        """

        filename = ''
        if data_category == 'group':
            filename = self.grp_file
        else:
            filename = self.part_file

        # All items in the snapshot file header are also included in the
        # group file header:
        with h5py.File(filename, 'r') as f:
            out = f['link0/{}'.format(entry)].attrs.get(attr)

        return out

    def link_select(self, data_category, fnums):
        """ Selects links from file keys and constructs an index list
        for sorting. """

        filename = ''
        if data_category == 'group':
            filename = self.grp_file
        else:
            filename = self.part_file

        with h5py.File(filename, 'r') as f:

            # Set to ndarray:
            keys = np.array(list(f.keys()))

            mask = [False for k in keys]
            for (idx, key) in enumerate(keys):
                if 'link' in key:
                    if len(fnums) == 0:
                        mask[idx] = True
                    elif int(key.replace('link', '')) in fnums:
                        mask[idx] = True

        keys = keys[mask]
        linknums = [int(key.replace('link', '')) for key in keys]
        sorting = np.argsort(linknums)

        return keys[sorting], sorting

    def convert_to_cgs_group(self, data, dataset):
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

        with h5py.File(self.grp_file, 'r') as grpf:
            # Get conversion factors.
            cgs = grpf['link1/Subhalo/{}'.format(dataset)].attrs \
                .get('CGSConversionFactor')
            aexp = grpf['link1/Subhalo/{}'.format(dataset)].attrs \
                .get('aexp-scale-exponent')
            hexp = grpf['link1/Subhalo/{}'.format(dataset)].attrs \
                .get('h-scale-exponent')

            # Get expansion factor and Hubble parameter from the 
            # header.
            a = grpf['link1/Header'].attrs.get('Time')
            h = grpf['link1/Header'].attrs.get('HubbleParam')

            # Convert to physical and return in cgs units.
            if data.dtype != np.int32 and data.dtype != np.int64:
                converted = np.multiply(data, cgs * a ** aexp * h ** hexp, \
                                        dtype='f8')

        return converted

    def convert_to_cgs_part(self, data, dataset):
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

        with h5py.File(self.part_file, 'r') as partf:
            # Get conversion factors (same for all types):
            cgs = partf['link1/PartType0/{}'.format(dataset)] \
                .attrs.get('CGSConversionFactor')
            aexp = partf['link1/PartType0/{}'.format(dataset)] \
                .attrs.get('aexp-scale-exponent')
            hexp = partf['link1/PartType0/{}'.format(dataset)] \
                .attrs.get('h-scale-exponent')

            # Get expansion factor and Hubble parameter from the header:
            a = partf['link1/Header'].attrs.get('Time')
            h = partf['link1/Header'].attrs.get('HubbleParam')

            # Convert to physical and return in cgs units.
            if data.dtype != np.int32 and data.dtype != np.int64:
                converted = np.multiply(data, cgs * a ** aexp * h ** hexp, \
                                        dtype='f8')

        return converted

    def convert_to_cgs_bound(self, data, dataset):
        """ Read conversion factors for a dataset of bound particles and 
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

        with h5py.File(self.grp_file, 'r') as grpf:
            # Get conversion factors (same for all types):
            cgs = grpf['link1/IDs/{}'.format(dataset)] \
                .attrs.get('CGSConversionFactor')
            aexp = grpf['link1/IDs/{}'.format(dataset)] \
                .attrs.get('aexp-scale-exponent')
            hexp = grpf['link1/IDs/{}'.format(dataset)] \
                .attrs.get('h-scale-exponent')

            # Get expansion factor and Hubble parameter from the header:
            a = grpf['link1/Header'].attrs.get('Time')
            h = grpf['link1/Header'].attrs.get('HubbleParam')

            # Convert to physical and return in cgs units.
            if data.dtype != np.int32 and data.dtype != np.int64:
                converted = np.multiply(data, cgs * a ** aexp * h ** hexp, \
                                        dtype='f8')

        return converted

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

        with h5py.File(self.grp_file, 'r') as grpf:

            links = [item for item in grpf.items() \
                     if ('link' in item[0])]

            for (name, link) in links:
                GNs = link['Subhalo/GroupNumber'][...]
                SGNs = link['Subhalo/SubGroupNumber'][...]

                if np.logical_and((GNs == gn), (SGNs == sgn)).sum() > 0:
                    fileNum = int(name.replace('link', ''))
                    break

        return fileNum
