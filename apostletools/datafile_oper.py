import numpy as np
import os
import h5py
from astropy import units
import re
import glob

import dataset_comp


class DataEnvelope:
    """

    Attributes
    ----------
    data_path : str
        Absolute path to the source data directory.
    fname : str
        Absolute path to the envelope file.

    """

    def __init__(self, data_files, env_fname, env_path=""):
        """

        Parameters
        ----------
        data_files : list of str
            Absolute paths to files that are enveloped.
        env_fname : str
            Envelope file name.
        env_path : str, optional
            Absolute path for the envelope file. Default is the
            ".ext_data_files" directory in the project home directory.
        """

        if not env_path:
            env_path = os.path.abspath(os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", ".ext_data_files"
            ))
        else:
            env_path = os.path.abspath(env_path)
        self.fname = os.path.join(env_path, env_fname)
        self.create_envelope_file(data_files)

    def create_envelope_file(self, files):
        """ Create an HDF5 file with links to files given as argument.

        Parameters
        ----------
        files : list of str
            Absolute paths to files that are enveloped.

        Notes
        -----
        The argument files is assumed to be ordered and the created links are
        in corresponding order, with their number labelling starting from 0.
        """

        # Create the file object with links to all the files:
        with h5py.File(self.fname, 'a') as f:

            # Iterate through data files and add missing links:
            for i, filename in enumerate(files):
                # Make an external link:
                if not 'link{}'.format(i) in f:
                    f['link{}'.format(i)] = \
                        h5py.ExternalLink(filename, '/')

    def save_dataset(self, dataset, ds_name, h5_group):
        # If the dataset already exists, replace it with the new data
        if self.dataset_exists(ds_name, h5_group):
            with h5py.File(self.fname, 'r+') as file:
                file['/{}/{}'.format(h5_group, ds_name)][...] = dataset
        # ...otherwise, create a new dataset:
        else:
            with h5py.File(self.fname, 'r+') as file:
                file.create_dataset('/{}/{}'.format(h5_group, ds_name),
                                    data=dataset)

    def dataset_exists(self, ds_name, h5_group):
        with h5py.File(self.fname, 'r') as f:
            if h5_group not in f:
                out = False
            else:
                out = ds_name in f[h5_group]
        return out


def create_dataset_in_group_envelope(snapshot, ds_name, h5_group):
    """ An interface to the functions of this module. Uses the correct
    function to construct the dataset.
    """

    out = np.array([])
    subgroup_list = str.split(h5_group, '/')

    # Datasets at the root of the "Extended" group:
    if len(subgroup_list) == 1:
        match_v_at_r = re.match("V([0-9]+)kpc", ds_name)
        if bool(match_v_at_r):
            r = int(match_v_at_r.groups()[0])
            out = dataset_comp.compute_vcirc(snapshot,
                                             r * units.kpc.to(units.cm))

        # DO NOT TRUST:
        elif ds_name == 'MassAccum':
            # Combine all particles from all subhalos into one long array.
            # Particles are ordered first by halo, then by particle
            # type, and lastly by distance to host halo.

            ma, r = dataset_comp.compute_mass_accumulation(
                snapshot, part_type=[0])
            for pt in [1, 4, 5]:
                ma_add, r_add = dataset_comp.compute_mass_accumulation(
                    snapshot, part_type=[pt])
                ma += ma_add
                r += r_add

            ma = np.concatenate(ma)
            r = np.concatenate(r)

            combined = np.column_stack((ma, r))

            out = combined

        elif ds_name == 'Max_Vcirc':
            vmax, rmax = dataset_comp.compute_vmax(snapshot)
            combined = np.column_stack((vmax, rmax))
            out = combined

        # Create dataset in the group data file:
        if out.size != 0:
            snapshot.group_data.save_dataset(out, ds_name, h5_group)

    elif subgroup_list[1] == 'RotationCurve':
        part_type = subgroup_list[2]
        pt_list = []
        if part_type == 'All':
            pt_list = [0, 1, 4, 5]
        else:
            pt_list = [int(part_type[-1])]

        v_circ, radii = dataset_comp.compute_rotation_curves(
            snapshot, n_soft=10, part_type=pt_list)

        # Save array lengths, and concatenate to single array (which is
        # HDF5 compatible):
        sub_length = np.array([r.size for r in radii])
        sub_offset = np.concatenate((np.array([0]),
                                     np.cumsum(sub_length)[:-1]))
        v_circ = np.concatenate(v_circ)
        radii = np.concatenate(radii)
        combined = np.column_stack((v_circ, radii))

        # Set return value:
        if ds_name == 'Vcirc':
            out = combined
        elif ds_name == 'SubOffset':
            out = sub_offset

        # Create datasets in the group data file:
        snapshot.group_data.save_dataset(combined, 'Vcirc', h5_group)
        snapshot.group_data.save_dataset(sub_offset, 'SubOffset', h5_group)

    return out


def get_path_to_sim(sim_id, path_to_snapshots=""):
    """ Constructs the path to the simulation data directory.

    Notes
    -----
    Assume directory structure (if path_to_snapshots is not given):
    /home
      /apostletools
        __file__
        [...]
      /snapshots
        /snapshot_...
          [particle data files...]
        /groups_...
          [subhalo data files...]
    """

    if not path_to_snapshots:
        home = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        )
        path = os.path.join(home, "snapshots", sim_id)
    else:
        path = os.path.join(path_to_snapshots, sim_id)

    return path


def get_group_data_path(sim_id, snap_id, sim_path=""):
    """ Constructs the path to the group data directory.

    Parameters
    ----------
    sim_id
    snap_id
    path_to_snapshots

    Returns
    -------
    path : str

    Notes
    -----
    The assumed directory naming convention is:
    "groups_[snapshot_ID]*", where in place of [snapshot_ID] there is a
    three digit integer.
    """

    if not sim_path:
        path_to_sim = get_path_to_sim(sim_id)
    else:
        path_to_sim = sim_path

    # Find the snapshot directory and add to path:
    for dir_name in os.listdir(path_to_sim):
        if "groups_{:03d}".format(snap_id) in dir_name:
            path = os.path.join(path_to_sim, dir_name)

    return path


def get_particle_data_path(sim_id, snap_id, sim_path=""):
    """ Constructs the path to the particle data directory.

    Parameters
    ----------
    sim_id
    snap_id
    path_to_snapshots

    Returns
    -------
    path : str

    Notes
    -----
    The assumed directory naming convention is:
    "snapshot_[snapshot_ID]*", where in place of [snapshot_ID] there is a
    three digit integer.
    """

    if not sim_path:
        path_to_sim = get_path_to_sim(sim_id)
    else:
        path_to_sim = sim_path

    # Find the snapshot directory and add to path:
    for dir_name in os.listdir(path_to_sim):
        if "snapshot_{:03d}".format(snap_id) in dir_name:
            path = os.path.join(path_to_sim, dir_name)

    return path


def get_group_data_files(sim_id, snap_id, sim_path=""):
    """

    Parameters
    ----------
    sim_id
    snap_id
    path_to_snapshots

    Returns
    -------
    files : ndarray of str
        Sorted array of the subhalo catalogue data file names.

    Notes
    -----
    Assumes the following naming convention for the data files:
    "eagle_subfind_tab*".
    """

    # Get data file names in an array:
    data_path = get_group_data_path(sim_id, snap_id, sim_path=sim_path)
    files = np.array(glob.glob(os.path.join(data_path, 'eagle_subfind_tab*')))

    # Sort in ascending order:
    fnum = [int(fname.split(".")[-2]) for fname in files]
    sorting = np.argsort(fnum)
    files = files[sorting]

    return files


def get_particle_data_files(sim_id, snap_id, sim_path=""):
    """

    Parameters
    ----------
    sim_id
    snap_id

    Returns
    -------
    files : ndarray of str
        Sorted array of the subhalo catalogue data file names.

    Notes
    -----
    Assumes the following naming convention for the data files:
    "snap*".
    """

    # Get data file names in an array:
    data_path = get_particle_data_path(sim_id, snap_id, sim_path=sim_path)
    files = np.array(glob.glob(os.path.join(data_path, 'snap*')))

    # Sort in ascending order:
    fnum = [int(fname.split(".")[-2]) for fname in files]
    sorting = np.argsort(fnum)
    files = files[sorting]

    return files


def get_snap_ids(sim_id, path_to_snapshots=""):
    """ Read the IDs of available snapshots in the simulation data.
    """

    path_to_sim = get_path_to_sim(sim_id, path_to_snapshots=path_to_snapshots)

    # Construct list of paths to each snapshot file:
    snapshot_paths = glob.glob(os.path.join(path_to_sim, "snapshot_*"))

    # Get file names without path and extension:
    fnames = [os.path.basename(os.path.normpath(path)) for path in
              snapshot_paths]

    # Get the snapshot identifiers (second item in the name) and sort:
    snap_ids = np.array([int(fname.split("_")[1]) for fname in fnames])
    snap_ids.sort()

    return snap_ids

