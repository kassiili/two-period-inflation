import numpy as np
import os
import h5py
from astropy import units

import dataset_compute


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
        for i,filename in enumerate(files):
            # Make an external link:
            if not 'link{}'.format(i) in f:
                f['link{}'.format(i)] = \
                        h5py.ExternalLink(filename, '/')


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

    home = os.path.dirname(os.path.realpath(__file__))
    if not path_to_snapshots:
        path = os.path.join(home, "snapshots", sim_id)
    else:
        path = os.path.join(path_to_snapshots, sim_id)

    prefix = ""
    if data_category == "part":
        prefix = "snapshot_"
    else:
        prefix += "groups_"

    # Find the snapshot directory and add to path:
    for dir_name in os.listdir(path):
        if "{}{:03d}".format(prefix, snap_id) in dir_name:
            path = os.path.join(path, dir_name)

    return path


def create_dataset(snapshot, dataset, group):
    """ An interface to the functions of this module. Uses the correct
    function to construct the dataset.
    """

    out = []
    subgroup_list = str.split(group, '/')
    if len(subgroup_list) == 1:
        if dataset == 'V1kpc':
            out = dataset_compute.compute_vcirc(snapshot,
                                                units.kpc.to(units.cm))

        elif dataset == 'MassAccum':
            # Combine all particles from all subhalos into one long array.
            # Particles are ordered first by halo, then by particle
            # type, and lastly by distance to host halo.

            ma, r = dataset_compute.compute_mass_accumulation(
                snapshot, part_type=[0])
            for pt in [1, 4, 5]:
                ma_add, r_add = dataset_compute.compute_mass_accumulation(
                    snapshot, part_type=[pt])
                ma += ma_add
                r += r_add

            ma = np.concatenate(ma)
            r = np.concatenate(r)

            combined = np.column_stack((ma, r))

            out = combined

        elif dataset == 'Max_Vcirc':
            vmax, rmax = dataset_compute.compute_vmax(snapshot)
            combined = np.column_stack((vmax, rmax))
            out = combined

        # Create dataset in grpf:
        with h5py.File(snapshot.grp_file, 'r+') as grpf:
            grpf.create_dataset('/{}/{}'.format(group, dataset), data=out)

    elif subgroup_list[1] == 'RotationCurve':
        part_type = subgroup_list[2]
        pt_list = []
        if part_type == 'All':
            pt_list = [0, 1, 4, 5]
        else:
            pt_list = [int(part_type[-1])]

        v_circ, radii = dataset_compute.compute_rotation_curves(
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
