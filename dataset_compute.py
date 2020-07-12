import numpy as np

from astropy import units
from astropy.constants import G


def get_satellites_by_group_number(snap, dataset, galaxy,
                                   split_luminous=False,
                                   prune_vmax=True):
    gns = snap.get_subhalos('GroupNumber')
    sgns = snap.get_subhalos('SubGroupNumber')

    # Extract satellites:
    mask_sat = np.logical_and(gns == galaxy, sgns != 0)

    if prune_vmax:
        maxpoint = snap.get_subhalos("Max_Vcirc", group="Extended")
        vmax = maxpoint[:, 0]
        mask_sat = np.logical_and(mask_sat, vmax > 0)

    if split_luminous:
        sm = snap.get_subhalos('Stars/Mass')
        mask_lum = (sm > 0)
        mask_dark = (sm == 0)
        satellites = {'luminous': dataset[np.logical_and(mask_sat,
                                                         mask_lum)],
                      'dark': dataset[np.logical_and(mask_sat,
                                                     mask_dark)]}
    else:
        satellites = dataset[mask_sat]

    return satellites


def get_satellites_by_distance(snap, dataset, galaxy, split_luminous=False,
                               prune_vmax=True, distance=300):
    gns = snap.get_subhalos('GroupNumber')
    sgns = snap.get_subhalos('SubGroupNumber')

    # Extract satellites:
    mask_sat = np.logical_and(gns == galaxy, sgns != 0)

    if prune_vmax:
        maxpoint = snap.get_subhalos("Max_Vcirc", group="Extended")
        vmax = maxpoint[:, 0]
        mask_sat = np.logical_and(mask_sat, vmax > 0)

    if split_luminous:
        sm = snap.get_subhalos('Stars/Mass')
        mask_lum = (sm > 0)
        mask_dark = (sm == 0)
        satellites = {'luminous': dataset[np.logical_and(mask_sat,
                                                         mask_lum)],
                      'dark': dataset[np.logical_and(mask_sat,
                                                     mask_dark)]}
    else:
        satellites = dataset[mask_sat]

    return satellites


def split_subhalos_by_group_number(snap, dataset, split_luminous=False,
                                   prune_vmax=True, m31=(1, 0), mw=(2, 0)):
    gns = snap.get_subhalos('GroupNumber')
    sgns = snap.get_subhalos('SubGroupNumber')

    # Extract MW and M31 satellites vs. isolated by group numbers:
    mask_sat = np.logical_and.reduce([np.logical_or(gns == m31[0],
                                                    gns == mw[0]),
                                      sgns != m31[1], sgns != mw[1],
                                      sgns != 0])
    mask_isol = np.logical_and(np.logical_or(gns != 1, gns != 2),
                               sgns == 0)

    if prune_vmax:
        maxpoint = snap.get_subhalos("Max_Vcirc", group="Extended")
        vmax = maxpoint[:, 0]
        mask_sat = np.logical_and(mask_sat, vmax > 0)
        mask_isol = np.logical_and(mask_isol, vmax > 0)

    if split_luminous:
        sm = snap.get_subhalos('Stars/Mass')
        mask_lum = (sm > 0)
        mask_dark = (sm == 0)
        satellites = {'luminous': dataset[np.logical_and(mask_sat,
                                                         mask_lum)],
                      'dark': dataset[np.logical_and(mask_sat,
                                                     mask_dark)]}
        isolated = {'luminous': dataset[np.logical_and(mask_isol,
                                                       mask_lum)],
                    'dark': dataset[np.logical_and(mask_isol,
                                                   mask_dark)]}
    else:
        satellites = dataset[mask_sat]
        isolated = dataset[mask_isol]

    split_data = {'satellites': satellites, 'isolated': isolated}

    return split_data


def split_subhalos_by_distance(snap, dataset, split_luminous=False,
                               prune_vmax=True, m31=(1, 0), mw=(2, 0)):
    cops = snap.get_subhalos("CentreOfPotential")
    M31_idx = snap.index_of_halo(m31[0], m31[1])
    MW_idx = snap.index_of_halo(mw[0], mw[1])
    M31_cop = cops[M31_idx]
    MW_cop = cops[MW_idx]

    # Compute distance to centrals:
    distance_to_M31 = np.linalg.norm(periodic_wrap(snap, M31_cop, cops) - \
                                     M31_cop, axis=1)
    distance_to_MW = np.linalg.norm(periodic_wrap(snap, MW_cop, cops) - \
                                    MW_cop, axis=1)

    # Compute distance to Local Group centre:
    LG_centre = compute_LG_centre(snap)
    distance_to_LG = np.linalg.norm(periodic_wrap(snap, LG_centre, cops) \
                                    - LG_centre, axis=1)

    # Extract MW and M31 satellites by distance (excluding M31 and MW):
    index = np.arange(np.size(cops, axis=0))
    max_dist_sat = 300 * units.kpc.to(units.cm)
    mask_sat_m31 = np.logical_and(distance_to_M31 < max_dist_sat,
                                  index != M31_idx)
    mask_sat_mw = np.logical_and(distance_to_MW < max_dist_sat,
                                  index != MW_idx)

    # Extract isolated galaxies, making sure they are well inside zoom
    # region:
    sgns = snap.get_subhalos('SubGroupNumber')
    mask_isol = np.logical_and.reduce([
        np.logical_not(mask_sat_m31),
        np.logical_not(mask_sat_mw),
        distance_to_LG < 2 * units.Mpc.to(units.cm),
        sgns == 0])

    if prune_vmax:
        maxpoint = snap.get_subhalos("Max_Vcirc", group="Extended")
        vmax = maxpoint[:, 0]
        mask_sat_m31 = np.logical_and(mask_sat_m31, vmax > 0)
        mask_sat_mw = np.logical_and(mask_sat_mw, vmax > 0)
        mask_isol = np.logical_and(mask_isol, vmax > 0)

    if split_luminous:
        sm = snap.get_subhalos('Stars/Mass')
        mask_lum = (sm > 0)
        mask_dark = (sm == 0)
        split_data = {'M31_satellites':
                        {'luminous': dataset[np.logical_and(mask_sat_m31,
                                                         mask_lum)],
                         'dark': dataset[np.logical_and(mask_sat_m31,
                                                        mask_dark)]},

                      'MW_satellites':
                          {'luminous': dataset[np.logical_and(mask_sat_mw,
                                                           mask_lum)],
                           'dark': dataset[np.logical_and(mask_sat_mw,
                                                          mask_dark)]},
                      'isolated':
                          {'luminous': dataset[np.logical_and(mask_isol,
                                                              mask_lum)],
                           'dark': dataset[np.logical_and(mask_isol,
                                                          mask_dark)]}}
    else:
        split_data= {'M31_satellites': dataset[mask_sat_m31],
                     'MW_satellites': dataset[mask_sat_mw],
                     'isolated': dataset[mask_isol]}

    return split_data


def compute_LG_centre(snap):
    # LG centre is in the middle between M31 and MW centres:
    cops = snap.get_subhalos("CentreOfPotential")
    M31_cop = cops[snap.index_of_halo(1, 0)]
    MW_cop = cops[snap.index_of_halo(2, 0)]
    LG_centre = (M31_cop + periodic_wrap(snap, M31_cop, MW_cop)) / 2

    return LG_centre


def split_satellites(snap, dataset, fnums=[]):
    """ Reads an attribute from snapshot and divides into satellites and
    isolated galaxies.
    
    Parameters
    ----------
    dataset : str
        attribute to be retrieved
    fnums : list of ints, optional
        Specifies files, which are to be read

    Returns
    -------
    data : tuple of HDF5 datasets
        Satellite data in the first entry and isolated galaxies data in
        the second.
    """

    sgns = snap.get_subhalos('SubGroupNumber', fnums=fnums)
    data = snap.get_subhalos(dataset, fnums=fnums)

    # Divide into satellites and isolated galaxies:
    data_sat = data[sgns != 0]
    data_isol = data[sgns == 0]

    return (data_sat, data_isol)


def compute_vcirc(snapshot, r):
    cmass, radii = compute_mass_accumulation(snapshot)

    n_parts_inside_r = [np.sum(np.array(radii_halo) < r) for radii_halo in
                        radii]

    # Exclude spurious cases:
    def condition(n, n_sub):
        return (n_sub < n) and (n_sub > 0)

    a = 2  # number of values averaged around r
    mass_inside_r = [np.mean(cm[n - int(a / 2):n - 1 + int(a / 2)])
                     if condition(len(cm), n) else 0
                     for cm, n in zip(cmass, n_parts_inside_r)]

    myG = G.to(units.cm ** 3 * units.g ** -1 * units.s ** -2).value
    v_circ_at_r = np.array([np.sqrt(m * myG / r) for m in mass_inside_r])

    return v_circ_at_r


def compute_rotation_curves(snapshot, n_soft=10, part_type=[0, 1, 4, 5]):
    """ Compute the smoothed rotation curves of all subhalos.

    Parameters
    ----------
    snapshot : Snapshot object
    n_soft : int, optional
        Number of particles summed over for a single point on the
        rotation curve.
    """
    cmass, radii = compute_mass_accumulation(snapshot, part_type=part_type)

    # Compute running average:
    radii = [np.array(r[n_soft::n_soft]) for r in radii]
    cmass = [np.array(cm[n_soft::n_soft]) for cm in cmass]

    myG = G.to(units.cm ** 3 * units.g ** -1 * units.s ** -2).value
    v_circ = [np.sqrt(cm * myG / r) for cm, r in zip(cmass, radii)]

    # Add zero:
    radii = np.array([np.concatenate((np.array([0]), r)) for r in radii])
    v_circ = np.array([np.concatenate((np.array([0]), v)) for v in v_circ])

    return v_circ, radii


def compute_vmax(snapshot, n_soft=5):
    v_circ, radii = compute_rotation_curves(snapshot, n_soft=n_soft)

    max_idx = [np.argmax(v) for v in v_circ]
    vmax = np.array([v[i] for v, i in zip(v_circ, max_idx)])
    rmax = np.array([r[i] for r, i in zip(radii, max_idx)])

    return vmax, rmax


def mass_accumulation_to_array(snapshot):
    sublentype = snapshot.get_subhalos('SubLengthType')
    splitting_points = np.cumsum(np.concatenate(sublentype))[:-1] \
        .astype(int)
    raw_cmass = snapshot.get_subhalos('MassAccum', group='Extended')
    cmass = raw_cmass[:, 0]
    radii = raw_cmass[:, 1]
    cmass = np.array(np.split(cmass, splitting_points)).reshape(
        (np.size(sublentype, axis=0), 6))
    radii = np.array(np.split(radii, splitting_points)).reshape(
        (np.size(sublentype, axis=0), 6))

    return cmass, radii


def compute_mass_accumulation(snapshot, part_type=[0, 1, 4, 5]):
    """ For each subhalo, compute the mass accumulation by radius.

    Parameters
    ----------
    snapshot
    part_type

    Returns
    -------
    cum_mass, grouped_radii : ndarray of ndarray

    Notes
    -----
    Only particles bound to a subhalo contribute to its mass.
    """

    # In order to not mix indices between arrays, we need all particle
    # arrays from grouping method:
    grouped_data = group_particles_by_subhalo(snapshot, 'Coordinates',
                                              'Masses',
                                              part_type=part_type)

    cops = snapshot.get_subhalos('CentreOfPotential')

    # Get particle radii from their host halo (wrapped):
    h = snapshot.get_attribute('HubbleParam', 'Header')
    boxs = snapshot.get_attribute('BoxSize', 'Header')
    boxs = snapshot.convert_to_cgs_group(boxs, 'CentreOfPotential') / h
    grouped_radii = [np.linalg.norm(
        np.mod(coords - cop + 0.5 * boxs, boxs) - 0.5 * boxs, axis=1)
        for coords, cop in zip(grouped_data['Coordinates'], cops)]

    # Sort particles, first by subhalo, then by distance from host:
    gns = np.concatenate(grouped_data['GroupNumber'])
    sgns = np.concatenate(grouped_data['SubGroupNumber'])
    radii = np.concatenate(grouped_radii)
    sort = np.lexsort((radii, sgns, gns))

    # Sort particle mass array:
    mass = np.concatenate(grouped_data['Masses'])
    part_num = [np.size(arr) for arr in grouped_data['Masses']]
    splitting_points = np.cumsum(part_num)[:-1]
    mass_split = np.split(mass[sort], splitting_points)

    # Sort also array of radii:
    grouped_radii = np.array([r for r in np.split(radii[sort],
                                                  splitting_points)])

    # Compute mass accumulation with radius for each subhalo:
    cum_mass = np.array([np.cumsum(mass) for mass in mass_split])

    return cum_mass, grouped_radii


def group_particles_by_subhalo(snapshot, *datasets,
                               part_type=[0, 1, 4, 5]):
    """ Get given datasets of bound particles and split them by host
    halo.

    Parameters
    ----------
    snapshot : Snapshot object
        Snapshot from which the datasets are retrieved.
    *datasets : list of str
        Names of the datasets to be retrieved and grouped.
    part_type : list of int, optional
        Specifies which particle types are retrieved.

    Returns
    -------
    grouped_data : dict
        A dictionary of the requested grouped datasets, with the names
        of the dataset as the keys.

    Notes
    -----
    The particles are sorted, first by group number of the host halo,
    then by its subgroup number. """

    # Get particle data:
    gns = snapshot.get_particles('GroupNumber', part_type=part_type)
    sgns = snapshot.get_particles('SubGroupNumber', part_type=part_type)
    grouped_data = {'GroupNumber': gns, 'SubGroupNumber': sgns}
    for dataset in datasets:
        grouped_data[dataset] = snapshot.get_particles(dataset,
                                                       part_type=part_type)

    # Get subhalo data:
    part_num = snapshot.get_subhalos('SubLengthType')[:,
               part_type].astype(int)

    # Exclude particles that are not bound to a subhalo:
    mask_bound = (sgns < np.max(gns))
    for key in grouped_data.keys():
        grouped_data[key] = grouped_data[key][mask_bound]

    # Sort particles first by group number then by subgroup number:
    sort = np.lexsort((grouped_data['SubGroupNumber'],
                       grouped_data['GroupNumber']))
    for key in grouped_data.keys():
        grouped_data[key] = grouped_data[key][sort]

    # Split particle data by halo:
    splitting_points = np.cumsum(np.sum(part_num, axis=1))[:-1]
    for key in grouped_data.keys():
        grouped_data[key] = np.split(grouped_data[key], splitting_points)

    return grouped_data


def periodic_wrap(snapshot, cop, coords):
    """ Account for the periodic boundary conditions by moving particles 
    to the periodic location, which is closest to the cop of their host
    halo. """

    h = snapshot.get_attribute('HubbleParam', 'Header')
    boxs = snapshot.get_attribute('BoxSize', 'Header')
    boxs = snapshot.convert_to_cgs_group(np.array([boxs]),
                                         'CentreOfPotential')[0] / h
    wrapped = np.mod(coords - cop + 0.5 * boxs, boxs) + cop - 0.5 * boxs

    return wrapped


def calculate_V1kpc_inProgress(snapshot):
    """ For each subhalo, calculate the circular velocity at 1kpc. 
    Assume that there are no jumps in the SubGroupNumber values in any
    of the groups."""

    # Get particle data:
    coords = snapshot.get_particles('Coordinates') \
             * units.cm.to(units.kpc)
    mass = snapshot.get_particle_masses() * units.g.to(units.Msun)

    # Get halo data:
    COPs = snapshot.get_subhalos('CentreOfPotential', \
                                 divided=False)[0] * units.cm.to(units.kpc)
    part_idx = get_subhalo_part_idx(snapshot)

    massWithin1kpc = np.zeros((COPs[:, 0].size))

    for idx, (cop, idx_list) in enumerate(zip(COPs, part_idx)):
        # Get coords and mass of the particles in the corresponding halo:
        halo_coords = coords[idx_list]
        halo_mass = mass[idx_list]

        # Calculate distances to COP:
        r = np.linalg.norm(halo_coords - cop, axis=1)

        # Get coordinates within 1kpc from COP:
        r1kpc_mask = np.logical_and(r > 0, r < 1)

        massWithin1kpc[idx] = halo_mass[r1kpc_mask].sum()

    myG = G.to(
        units.km ** 2 * units.kpc * units.Msun ** -1 * units.s ** -2).value
    v1kpc = np.sqrt(massWithin1kpc * myG)

    return v1kpc


def get_subhalo_part_idx(snapshot):
    """ Finds indices of the particles in each halo. """

    # Get subhalos:
    halo_gns = snapshot.get_subhalos('GroupNumber', \
                                     divided=False)[0].astype(int)
    halo_sgns = snapshot.get_subhalos('SubGroupNumber', \
                                      divided=False)[0].astype(int)

    # Get particles:
    part_gns = snapshot.get_particles('GroupNumber')
    part_sgns = snapshot.get_particles('SubGroupNumber')

    # Get halo indices:
    sorting = np.lexsort((halo_sgns, halo_gns))
    print(halo_gns.size)

    # Invert sorting:
    inv_sorting = [0] * len(sorting)
    for idx, val in enumerate(sorting):
        inv_sorting[val] = idx

    # Loop through particles and save indices to lists. Halos in the
    # list behind the part_idx key are arranged in ascending order 
    # with gn and sgn, i.e. in the order lexsort would arrange them:
    gn_count = np.bincount(halo_gns)
    print(halo_gns.size + sum(gn_count == 0))
    print(gn_count.sum() + sum(gn_count == 0))
    part_idx = [[] for i in range(halo_gns.size + sum(gn_count == 0))]
    for idx, (gn, sgn) in enumerate(zip(part_gns, part_sgns)):
        # Exclude unbounded particles (for unbounded: sgn = max_int):
        if sgn < 10 ** 6:
            i = gn_count[:gn].sum() + sgn
            if i >= len(part_idx):
                print(i)
                print(gn, sgn)
            part_idx[i].append(idx)

    print(len(part_idx))
    # Strip from empty lists:
    part_idx = [l for l in part_idx if not (not l)]
    print(len(part_idx))

    # Convert to ndarray and sort in order corresponding to the halo
    # datasets:
    part_idx = np.array(part_idx)[inv_sorting]

    return part_idx
