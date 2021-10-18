import numpy as np
import os
import h5py
from astropy import units
import re
import glob

import dataset_comp


def path_to_extended(path_from_home=".ext_data_files"):
    """ Return the path to the extended data files.

    Returns
    -------
    ext_dir : str
        Path to the directory that contains all extended data files.

    Notes
    -----
    The extended data files are HDF5 files that collect links to all snapshot
    files of a given snapshot and work as storage for dataset extensions.
    """

    home = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    )

    ext_dir = os.path.join(home, path_from_home)

    return ext_dir


def create_common_group_file(sim_id, snap_id):
    """ Combine all group data files of a snapshot into a single HDF5 file. """

    # Set the file path and name:
    grp_file = os.path.join(
        path_to_extended(),
        ".groups_{}_{:03d}.hdf5".format(sim_id, snap_id)
    )

    data_path = get_data_path('group', sim_id, snap_id)
    files = np.array(glob.glob(os.path.join(data_path, 'eagle_subfind_tab*')))

    combine_data_files(files, grp_file)

    return grp_file


def create_common_part_file(sim_id, snap_id):
    """ Combine all particle data files of a snapshot into a single HDF5
    file. """

    # Set the file path and name:
    part_file = os.path.join(
        path_to_extended(),
        ".particles_{}_{:03d}.hdf5".format(sim_id, snap_id)
    )

    data_path = get_data_path('part', sim_id, snap_id)
    files = np.array(glob.glob(os.path.join(data_path, 'snap*')))

    combine_data_files(files, part_file)

    return part_file


def combine_data_files(files, filename):
    """ Create an HDF5 file object and add links to all given files

    Parameters
    ----------
    files : str
        path to files
    filename : str
        name of the combined file
    """

    # Sort in ascending order:
    fnum = [int(fname.split(".")[-2]) for fname in files]
    sorting = np.argsort(fnum)
    files = files[sorting]

    # Create the file object with links to all the files:
    with h5py.File(filename, 'a') as f:

        # Iterate through data files and add missing links:
        for i, filename in enumerate(files):
            # Make an external link:
            if not 'link{}'.format(i) in f:
                f['link{}'.format(i)] = \
                    h5py.ExternalLink(filename, '/')


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


def get_data_path(data_category, sim_id, snap_id, path_to_snapshots=""):
    """ Constructs the path to data directory.

    Paramaters
    ----------
    data_category : str
        recognized values are: 'part' and 'group'

    Returns
    -------
    path : str
        path to data directory
    """

    path_to_sim = get_path_to_sim(sim_id, path_to_snapshots=path_to_snapshots)

    prefix = ""
    if data_category == "part":
        prefix = "snapshot_"
    else:
        prefix += "groups_"

    # Find the snapshot directory and add to path:
    for dir_name in os.listdir(path_to_sim):
        if "{}{:03d}".format(prefix, snap_id) in dir_name:
            path = os.path.join(path_to_sim, dir_name)

    return path


def group_dataset_exists(snapshot, dataset, h5_group):
    with h5py.File(snapshot.grp_file, 'r') as f:
        if h5_group not in f:
            out = False
        else:
            out = dataset in f[h5_group]
    return out


def get_snap_ids(sim_id, path_to_snapshots=""):
    """ Read the snapshot identifiers of snapshots in a simulation.
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


def create_dataset(snapshot, dataset, group):
    """ An interface to the functions of this module. Uses the correct
    function to construct the dataset.
    """

    out = np.array([])
    subgroup_list = str.split(group, '/')

    # Datasets at the root of the "Extended" group:
    if len(subgroup_list) == 1:
        match_v_at_r = re.match("V([0-9]+)kpc", dataset)
        if bool(match_v_at_r):
            r = int(match_v_at_r.groups()[0])
            out = dataset_comp.compute_vcirc(snapshot,
                                             r * units.kpc.to(
                                                    units.cm))

        # DO NOT TRUST:
        elif dataset == 'MassAccum':
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

        elif dataset == 'Max_Vcirc':
            vmax, rmax = dataset_comp.compute_vmax(snapshot)
            combined = np.column_stack((vmax, rmax))
            out = combined

        # Create dataset in grpf:
        if out.size != 0:
            with h5py.File(snapshot.grp_file, 'r+') as grpf:
                grpf.create_dataset('/{}/{}'.format(group, dataset),
                                    data=out)

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
        if dataset == 'Vcirc':
            out = combined
        elif dataset == 'SubOffset':
            out = sub_offset

        # Create datasets in grpf:
        with h5py.File(snapshot.grp_file, 'r+') as grpf:
            grpf.create_dataset('/{}/Vcirc'.format(group), data=combined)
            grpf.create_dataset('/{}/SubOffset'.format(group),
                                data=sub_offset)

    return out


def save_dataset(data, dataset, group, snapshot):
    # If the dataset already exists, replace it with the new data
    if group_dataset_exists(snapshot, dataset, group):
        with h5py.File(snapshot.grp_file, 'r+') as grpf:
            grpf['/{}/{}'.format(group, dataset)][...] = data
    # ...otherwise, create a new dataset:
    else:
        with h5py.File(snapshot.grp_file, 'r+') as grpf:
            grpf.create_dataset('/{}/{}'.format(group, dataset),
                                data=data)
