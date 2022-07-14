import os
import glob
import numpy as np
import h5py

import datafile_oper


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
    group_data : DataEnvelope object
        Envelope for all subhalo data.
    part_data : DataEnvelope object
        Envelope for all particle data.
    """

    def __init__(self, sim_id, snap_id, name="", sim_path="", grp_env="",
                 part_env="", env_path=""):
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
        sim_path : str, optional
            Path to the simulation data directory.
        grp_env : str
            Group data envelope file name.
        part_env : str, optional
            Particle data envelope file name.
        env_path : str, optional
            Absolute path for the envelope files.
        """

        self.sim_id = sim_id
        self.snap_id = snap_id

        # If not given, construct name from IDs:
        if not name:
            self.name = "{}_{:03d}".format(sim_id, snap_id)
        else:
            self.name = name
        if not grp_env:
            grp_env = ".groups_{}_{:03d}.hdf5".format(sim_id, snap_id)
        if not part_env:
            part_env = ".particles_{}_{:03d}.hdf5".format(sim_id, snap_id)

        self.group_data = datafile_oper.DataEnvelope(
            datafile_oper.get_group_data_files(sim_id, snap_id,
                                               sim_path=sim_path),
            grp_env,
            env_path
        )
        self.part_data = datafile_oper.DataEnvelope(
            datafile_oper.get_particle_data_files(sim_id, snap_id,
                                                  sim_path=sim_path),
            part_env,
            env_path
        )

    def get_subhalos(self, dataset, h5_group='Subhalo', units='cgs'):
        """ Retrieves a dataset for subhaloes in the snapshot.

        Parameters
        ----------
        dataset : str
            Name of dataset to be retrieved.
        h5_group : str, optional
            Name of the group that contains the dataset.

        Returns
        -------
        out : HDF5 datasets
            The requested dataset.
        """

        # Output array.
        out = np.empty(0)

        # Check whether dataset is in the catalogues or an extension:
        if str.split(h5_group, '/')[0] == 'Extended':
            # Check if the dataset is already stored in grpf:
            in_grpf = False
            with h5py.File(self.group_data.fname, 'r') as grpf:
                if '{}/{}'.format(h5_group, dataset) in grpf:
                    out = grpf['{}/{}'.format(h5_group, dataset)][...]
                    in_grpf = True

            if not in_grpf:
                out = datafile_oper.create_dataset_in_group_envelope(
                    self, dataset, h5_group
                )

        else:
            out = self.get_subhalo_catalogue(dataset, h5_group, units)

        return out

    def get_subhalo_catalogue_old(self, dataset, group, fnums=None, units='cgs',
                              comov=False):
        """ Retrieves a dataset from the subhalo catalogues.

        Parameters
        ----------
        dataset : str
            Name of dataset to be retrieved.
        group : str
            Name of the HDF5 group, which encloses dataset.
        fnums : list of ints, optional
            Specifies files, which are to be read.

        Returns
        -------
        out : HDF5 dataset
            Requested dataset in cgs units.
        """

        if fnums is None:
            fnums = []

        out = []
        link_names, link_sort = self.link_select('group', fnums)

        with h5py.File(self.group_data.fname, 'r') as grpf:

            links = [f for (name, f) in grpf.items()
                     if name in link_names]
            for f in links:
                tmp = f['{}/{}'.format(group, dataset)][...]
                out.append(tmp)

        # Sort by link number:
        out = [out[i] for i in link_sort]

        # Combine to a single array.
        if len(out[0].shape) > 1:
            out = np.vstack(out)
        else:
            out = np.concatenate(out)

        if units == 'cgs':
            out = self.convert_to_cgs_group(out, dataset, group)

        return out


    def get_subhalo_catalogue(self, dataset, group, units='cgs', comov=False):
        """ Retrieves a dataset from the subhalo catalogues.

        Parameters
        ----------
        dataset : str
            Name of dataset to be retrieved.
        group : str
            Name of the HDF5 group, which encloses dataset.

        Returns
        -------
        out : HDF5 dataset
            Requested dataset in cgs units.
        """

        out = []
        link_names, link_sort = self.link_select(
            self.group_data.fname, dataset, group
        )

        with h5py.File(self.group_data.fname, 'r') as grpf:

            links = [f for (name, f) in grpf.items()
                     if name in link_names]
            for f in links:
                tmp = f['{}/{}'.format(group, dataset)][...]
                out.append(tmp)

        # Sort by link number:
        out = [out[i] for i in link_sort]

        # Combine to a single array.
        if len(out[0].shape) > 1:
            out = np.vstack(out)
        else:
            out = np.concatenate(out)

        if units == 'cgs':
            out = self.convert_to_cgs_group(out, dataset, group)

        return out

    def get_subhalos_IDs(self, part_type=[0, 1, 4, 5]):
        """ Read IDs of bound particles of given type for each halo.
        
        Paramaters
        ----------
        part_type : list of int
            Types of particles, whose attribute values are retrieved (the
            default is [0, 1, 4, 5], i.e. all high-res particle types)
        fnums : list of ints, optional
            Specifies files, which are to be read

        Returns
        -------
        IDs : HDF5 dataset 
            Dataset of ndarrays of bound particles
        """

        IDs_bound = self.get_bound_particles("ParticleID")

        # Construct mask for selecting bound particles of type pt:
        IDs_pt = self.get_particles("ParticleIDs", part_type=part_type)
        mask_pt = np.isin(IDs_bound, IDs_pt)

        IDs = []
        link_names, link_sort = self.link_select(
            self.group_data.fname, 'SubOffset', 'Subhalo'
        )
        with h5py.File(self.group_data.fname, 'r') as grpf:
            # Get IDs by halo from selected files:
            links = [f for (name, f) in grpf.items() \
                     if name in link_names]
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
            IDs = [IDs[i] for i in link_sort]
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
        link_names, link_sort = self.link_select(
            self.group_data.fname, dataset, 'IDs'
        )

        with h5py.File(self.group_data.fname, 'r') as grpf:
            # Get particle IDs from all files:
            out = []
            links = [f for (name, f) in grpf.items() \
                     if name in link_names]

            for link in links:
                out.append(link['IDs/{}'.format(dataset)][...])

        # Sort by link number:
        out = [out[i] for i in link_sort]
        out = np.concatenate(out)

        out = self.convert_to_cgs_bound(out, dataset)

        return out

    def get_subhalo_number(self):
        """ Return the number of subhalos in this snapshot. """
        return self.get_attribute("TotNsubgroups", "Header")

    def get_particles(self, dataset, part_type=None):
        """ Reads the dataset from particle catalogues.
        
        Parameters
        ----------
        dataset : str
            Dataset to be retrieved.
        part_type : list of int, optional
            Types of particles, whose attribute values are retrieved (the
            default is [0, 1, 4, 5], i.e. all high-res particle types)

        Returns
        -------
        out : HDF5 dataset
            Requested dataset in cgs units.
        """

        if part_type is None:
            part_type = [0, 1, 4, 5]

        out = []
        if dataset == 'Masses':
            out = self.get_particle_masses(part_type)
        elif dataset == 'PartType':
            out = self.get_particle_types(part_type=part_type)
        else:
            out = self.get_particle_catalogue(dataset, part_type=part_type)

        return out

    def get_particle_types(self, part_type=None):
        """ Constructs an array of particle types of particles in the
        order of the catalogue method. """

        if part_type is None:
            part_type = [0, 1, 4, 5]

        # Read total number of particles of each type:
        num_part_tot = self.get_attribute("NumPart_Total", "Header",
                                          "particle")

        # Construct an array indicating particle types of all particles:
        out = np.concatenate([pt * np.ones(num_part_tot[pt])
                              for pt in part_type])

        return out

    def get_particle_catalogue(self, dataset, part_type=[0, 1, 4, 5]):
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

        link_names = {}
        link_sort = {}
        for pt in part_type:
            link_names[pt], link_sort[pt] = self.link_select(
                self.part_data.fname, dataset, 'PartType{}'.format(pt)
            )

        # Get particle file:
        with h5py.File(self.part_data.fname, 'r') as partf:

            # For each particle type, loop over files, s.t. elements in
            # out are primarily ordered by particle type:
            for pt in part_type:

                links = [f for (name, f) in partf.items()
                         if name in link_names[pt]]
                links = [links[i] for i in link_sort[pt]]
                for f in links:
                    if 'PartType{}/{}'.format(pt, dataset) in f:
                        tmp = f['PartType{}/{}'.format(pt, dataset)][...]
                        out.append(tmp)

        # Combine to a single array.
        if len(out[0].shape) > 1:
            out = np.vstack(out)
        else:
            out = np.concatenate(out)

        out = self.convert_to_cgs_part(
            out, dataset, "PartType{}".format(part_type[0])
        )

        return out

    def get_particle_masses(self, part_type=[0, 1, 4, 5]):
        """ Reads particle masses in grams.
        
        Returns
        -------
        mass : HDF5 dataset
            Masses of each particle in cgs.
        """

        link_names, _ = self.link_select(self.group_data.fname, "Header", "")
        link = link_names[0]

        mass = []

        for pt in part_type:
            if pt in [1]:
                # Get dm particle masses:
                with h5py.File(self.part_data.fname, 'r') as partf:
                    dm_mass = partf['{}/Header'.format(link)] \
                        .attrs.get('MassTable')[pt]
                    dm_mass = self.convert_to_cgs_part(
                        np.array([dm_mass]), 'Masses', 'PartType0')[0]
                    dm_n = partf['{}/Header'.format(link)] \
                        .attrs.get('NumPart_Total')[pt]
                    mass.append(np.ones(dm_n, dtype='f8') * dm_mass)
            else:
                mass.append(self.get_particle_catalogue('Masses',
                                                        part_type=[pt]))

        mass = np.concatenate(mass)

        return mass

    def get_attribute(self, attr, entry, data_category='subhalo'):
        """ Reads an attribute of the given entry (either dataset or group).

        Parameters
        ----------
        attr : str
            Attribute name.
        entry : str
            Group or dataset name with the attribute.
        data_category : str, optional
            Specifies, in which datafile the given attribute is read from.

        Returns
        -------
        out : float
            The value of the attribute.
        """

        if data_category == 'subhalo':
            filename = self.group_data.fname
        else:
            filename = self.part_data.fname

        # Check whether dataset is in the catalogues or an extension:
        if str.split(entry, '/')[0] == 'Extended':
            with h5py.File(filename, 'r') as f:
                out = f[entry].attrs.get(attr)
        else:
            # Get links that contain entry:
            link_names,_ = self.link_select(filename, entry, "")

            # All items in the snapshot file header are also included in the
            # group file header:
            with h5py.File(filename, 'r') as f:
                out = f['{}/{}'.format(link_names[0], entry)].attrs.get(attr)

        return out

    def link_select(self, filename, dset_name, h5_group):
        """ Selects links from file keys and constructs an index list
        for sorting.

        Notes
        -----
        Assumes source data links have names of the form 'link#', where #
        stands for the link number.
        """

        with h5py.File(filename, 'r') as f:

            # Get links to source data files, in the given envelope file:
            link_names = np.array([
                key for key in f.keys() if ('link' in key)
            ])

            # Select links that have the given dataset:
            link_names = np.array([
                link for link in link_names
                if ("{}/{}".format(h5_group, dset_name) in f[link])
            ])

        link_nums = [int(link.replace('link', '')) for link in link_names]
        sorting = np.argsort(link_nums)

        return link_names[sorting], sorting

    def link_select_old(self, data_category, fnums):
        """ Selects links from file keys and constructs an index list
        for sorting. """

        filename = ''
        if data_category == 'group':
            filename = self.group_data.fname
        else:
            filename = self.part_data.fname

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

    def convert_to_cgs_group(self, data, dataset, group='Subhalo'):
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

        link_names, _ = self.link_select(self.group_data.fname, dataset, group)
        link = link_names[0]

        with h5py.File(self.group_data.fname, 'r') as grpf:
            # Get conversion factors.
            cgs = grpf['{}/{}/{}'.format(link, group, dataset)].attrs \
                .get('CGSConversionFactor')
            aexp = grpf['{}/{}/{}'.format(link, group, dataset)].attrs \
                .get('aexp-scale-exponent')
            hexp = grpf['{}/{}/{}'.format(link, group, dataset)].attrs \
                .get('h-scale-exponent')

            # Get expansion factor and Hubble parameter from the 
            # header.
            a = grpf['{}/Header'.format(link)].attrs.get('Time')
            h = grpf['{}/Header'.format(link)].attrs.get('HubbleParam')

            # Convert to physical and return in cgs units.
            if data.dtype != np.int32 and data.dtype != np.int64:
                converted = np.multiply(data, cgs * a ** aexp * h ** hexp,
                                        dtype='f8')

        return converted

    def convert_to_cgs_part(self, data, dset_name, h5_group):
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

        link_names, _ = self.link_select(self.part_data.fname, dset_name,
                                         h5_group)
        link = link_names[0]

        with h5py.File(self.part_data.fname, 'r') as partf:
            # Get conversion factors (same for all types):
            cgs = partf['{}/{}/{}'.format(link, h5_group, dset_name)].attrs \
                .get('CGSConversionFactor')
            aexp = partf['{}/{}/{}'.format(link, h5_group, dset_name)].attrs \
                .get('aexp-scale-exponent')
            hexp = partf['{}/{}/{}'.format(link, h5_group, dset_name)].attrs \
                .get('h-scale-exponent')

            # Get expansion factor and Hubble parameter from the header:
            a = partf['{}/Header'.format(link)].attrs.get('Time')
            h = partf['{}/Header'.format(link)].attrs.get('HubbleParam')

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

        link_names, _ = self.link_select(self.group_data.fname, dataset, "IDs")
        link = link_names[0]

        with h5py.File(self.group_data.fname, 'r') as grpf:
            # Get conversion factors (same for all types):
            cgs = grpf['{}/IDs/{}'.format(link, dataset)] \
                .attrs.get('CGSConversionFactor')
            aexp = grpf['{}/IDs/{}'.format(link, dataset)] \
                .attrs.get('aexp-scale-exponent')
            hexp = grpf['{}/IDs/{}'.format(link, dataset)] \
                .attrs.get('h-scale-exponent')

            # Get expansion factor and Hubble parameter from the header:
            a = grpf['{}/Header'.format(link)].attrs.get('Time')
            h = grpf['{}/Header'.format(link)].attrs.get('HubbleParam')

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

        with h5py.File(self.group_data.fname, 'r') as grpf:

            links = [item for item in grpf.items() \
                     if ('link' in item[0])]

            for (name, link) in links:
                GNs = link['Subhalo/GroupNumber'][...]
                SGNs = link['Subhalo/SubGroupNumber'][...]

                if np.logical_and((GNs == gn), (SGNs == sgn)).sum() > 0:
                    fileNum = int(name.replace('link', ''))
                    break

        return fileNum

    def index_of_halo(self, gn, sgn):

        gns = self.get_subhalos("GroupNumber")
        sgns = self.get_subhalos("SubGroupNumber")
        idx = np.nonzero(np.logical_and(gns == gn, sgns == sgn))[0][0]

        return idx
