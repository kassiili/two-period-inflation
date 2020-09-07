import numpy as np

from astropy import units
from astropy import constants


def split_luminous(snap):
    sm = snap.get_subhalos('Stars/Mass')
    mask_lum = (sm > 0)
    mask_dark = (sm == 0)
    return mask_lum, mask_dark


def prune_vmax(snap, low_lim=10**(-10), up_lim=10**10):
    maxpoint = snap.get_subhalos("Max_Vcirc", group="Extended")
    vmax = maxpoint[:, 0]
    low_lim = low_lim * units.km.to(units.cm)
    up_lim = up_lim * units.km.to(units.cm)
    mask = np.logical_and(vmax >= low_lim, vmax < up_lim)
    return mask


def split_satellites_by_r200(snap, m31_ident, mw_ident,
                             max_dist_isol=2000):
    r200_m31 = snap.get_subhalos("Group_R_Mean200", "FOF")[m31_ident[0]] \
               * units.cm.to(units.kpc)
    r200_mw = snap.get_subhalos("Group_R_Mean200", "FOF")[mw_ident[0]] \
              * units.cm.to(units.kpc)

    return split_satellites_by_distance(snap, m31_ident, mw_ident,
                                        max_dist_sat=[r200_m31, r200_mw],
                                        max_dist_isol=max_dist_isol)


def split_satellites_by_distance(snap, m31_ident, mw_ident,
                                 max_dist_sat=300,
                                 max_dist_isol=2000):
    """ Select satellites and isolated galaxies from subhalos by their
    distance to the M31 and MW galaxies and the LG barycentre,
    respectively.

    Parameters
    ----------
    snap : Snapshot object
    m31_ident : tuple of two int
        Group number and subgroup number of the M31 halo.
    mw1_ident : tuple of two int
        Group number and subgroup number of the MW halo.
    max_dist_sat : float, optional
        Maximum distance for satellite galaxies in units of kpc. Default
        value is 300 kpc.
    max_dist_isol : float, optional
        Maximum distance for isolated galaxies in units of kpc. Default
        value is 2000 kpc.

    Returns
    -------
    masks_sat, mask_isol : ndarray of bool

    Notes
    -----
    Satellites of a central halo are defined as subhalos lying within a
    distance (like 300kpc) from the central. In case the satellites of the
    two given centrals intersect, they are assigned to their hosts by
    distance. Isolated galaxies are defined as the subhalos that are not
    bound to another halo (i.e. have subgroup number == 0), and lie within
    a distance from the barycentre of the two centrals but are not
    satellites.
    """

    # If max_dist_sat is not a list, assume it is a float (or int):
    if not isinstance(max_dist_sat, list):
        max_dist_sat = [max_dist_sat, max_dist_sat]

    centrals = [m31_ident, mw_ident]
    cops = snap.get_subhalos("CentreOfPotential")
    gns = snap.get_subhalos("GroupNumber")
    sgns = snap.get_subhalos("SubGroupNumber")

    # Select satellites:
    dist_to_centrals = [distance_to_point(
        snap, cops[snap.index_of_halo(c[0], c[1])])
        for c in centrals]
    min_dist = 0.01 * units.kpc.to(units.cm)
    max_dist_sat = [d * units.kpc.to(units.cm) for d in max_dist_sat]
    masks_sat = [within_distance_range(d, min_dist, max_dist)
                 for d, max_dist in zip(dist_to_centrals, max_dist_sat)]

    # Find intersection and split by distance to centrals:
    mask_intersect = np.logical_and.reduce(masks_sat)
    mask_closer_to0 = dist_to_centrals[0] >= dist_to_centrals[1]
    mask_intersect_split = [
        np.logical_and(mask_intersect, mask_closer_to0),
        np.logical_and(mask_intersect,
                       np.logical_not(mask_closer_to0))
    ]

    # Remove satellites closer to the other halo:
    masks_sat[0] = np.logical_and(
        masks_sat[0], np.logical_not(mask_intersect_split[1]))
    masks_sat[1] = np.logical_and(
        masks_sat[1], np.logical_not(mask_intersect_split[0]))

    # Select isolated galaxies:
    dist_to_lg = distance_to_point(snap,
                                   compute_LG_centre(snap, m31_ident,
                                                     mw_ident))
    max_dist_isol = max_dist_isol * units.kpc.to(units.cm)
    mask_isol = within_distance_range(dist_to_lg, 0, max_dist_isol)
    exclude_centrals = [np.logical_not(np.logical_and(
        gns == c_any[0], sgns == c_any[1])) for c_any in centrals]
    mask_isol = np.logical_and.reduce([mask_isol,
                                       np.logical_not(masks_sat[0]),
                                       np.logical_not(masks_sat[1]),
                                       sgns == 0] +
                                       exclude_centrals)
    return masks_sat, mask_isol


def split_satellites_by_group_number(snap, *centrals):
    gns = snap.get_subhalos("GroupNumber")
    sgns = snap.get_subhalos("SubGroupNumber")

    masks_sat = []
    for c in centrals:
        if c[1] == 0:
            exclude_centrals = [np.logical_not(np.logical_and(
                gns == c_any[0], sgns == c_any[1])) for c_any in centrals]
            masks_sat.append(np.logical_and.reduce(
                exclude_centrals + [sgns != 0, gns == c[0]]))
        else:
            masks_sat.append(gns == -gns)

    mask_isol = np.logical_and.reduce([gns != c[0] for c in centrals] +
                                      [sgns == 0])

    return masks_sat, mask_isol


def within_distance_range(dist, min_r, max_r):
    """ Find subhalos within a radius of a given subhalo.

    Returns
    -------
    mask_within_radius, dist : ndarray of bool, ndarray of float
        Mask array of subhalos within a given distance, along with
        distances to the given subhalo.

    """
    mask_within_radius = np.logical_and(dist < max_r, dist > min_r)

    return mask_within_radius


def distance_to_subhalo(snap, gn, sgn):
    cop = snap.get_subhalos("CentreOfPotential")[snap.index_of_halo(gn, sgn)]
    return distance_to_point(snap, cop)

def distance_to_point(snap, point):
    """ For all halos in a snapshot, compute distnace to a given point. """
    cops = snap.get_subhalos("CentreOfPotential")
    dist = np.linalg.norm(periodic_wrap(snap, point, cops) - point,
                          axis=1)
    return dist


def compute_LG_centre(snap, m31_ident, mw_ident):
    # LG centre is in the middle between M31 and MW centres:
    cops = snap.get_subhalos("CentreOfPotential")
    m31_cop = cops[snap.index_of_halo(m31_ident[0], m31_ident[1])]
    mw_cop = cops[snap.index_of_halo(mw_ident[0], mw_ident[1])]
    LG_centre = (m31_cop + periodic_wrap(snap, m31_cop, mw_cop)) / 2

    return LG_centre


def compute_vcirc(snapshot, r):
    """ Compute subhalo circular velocities at given radius. """
    print("Computing v_circ at {} kpc for {}...".format(r, snapshot.sim_id))
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

    myG = constants.G.to(units.cm ** 3 * units.g ** -1 * units.s **
                         -2).value
    v_circ_at_r = np.array([np.sqrt(m * myG / r) for m in mass_inside_r])

    print("Done.")
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
    print("Computing subhalo rotation curves for {}...".format(snapshot.sim_id))
    cmass, radii = compute_mass_accumulation(snapshot, part_type=part_type)

    # Compute running average:
    radii = [np.array(r[n_soft - 1::n_soft]) for r in radii]
    cmass = [np.array(cm[n_soft - 1::n_soft]) for cm in cmass]

    myG = constants.G.to(units.cm ** 3 * units.g ** -1 * units.s **
                         -2).value
    v_circ = [np.sqrt(cm * myG / r) for cm, r in zip(cmass, radii)]

    # Add zero:
    radii = np.array([np.concatenate((np.array([0]), r)) for r in radii])
    v_circ = np.array(
        [np.concatenate((np.array([0]), v)) for v in v_circ])

    print("Done.")
    return v_circ, radii


def compute_vmax(snapshot, n_soft=5):
    """ Compute subhalo maximum circular velocity. """
    v_circ, radii = compute_rotation_curves(snapshot, n_soft=n_soft)

    max_idx = [np.argmax(v) for v in v_circ]
    vmax = np.array([v[i] for v, i in zip(v_circ, max_idx)])
    rmax = np.array([r[i] for r, i in zip(radii, max_idx)])

    return vmax, rmax


# Perhaps useless?
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
    grouped_data = group_particles_by_subhalo(
        snapshot, 'Coordinates', 'Masses', part_type=part_type)

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
    grouped_radii = [r for r in np.split(radii[sort], splitting_points)]

    # Compute mass accumulation with radius for each subhalo:
    cum_mass = [np.cumsum(mass) for mass in mass_split]

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


def group_selected_particles_by_subhalo(snapshot, *datasets,
                                        selection_mask=None,
                                        part_type=None):
    """ Get given datasets of bound particles, split them by host halo,
    and order by particle type.

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

    if part_type is None:
        part_type = [0, 1, 4, 5]

    if selection_mask is None:
        part_num = np.sum(
            snapshot.get_attribute("NumPart_Total", "Header",
                                   "particle")[part_type])
        selection_mask = np.ones(part_num)

    # Get particle data:
    gns = snapshot.get_particles('GroupNumber', part_type=part_type)
    sgns = snapshot.get_particles('SubGroupNumber', part_type=part_type)
    ptypes = snapshot.get_particles('PartType', part_type=part_type)

    # Exclude particles that are not bound to a subhalo:
    mask = np.logical_and(selection_mask, sgns < np.max(gns))
    gns = gns[mask]
    sgns = sgns[mask]
    ptypes = ptypes[mask]

    sort, splitting_points = sort_and_split_by_subhalo(snapshot, gns,
                                                       sgns,
                                                       add_sort=[ptypes])

    grouped_data = {'GroupNumber': np.split(gns[sort], splitting_points),
                    'SubGroupNumber': np.split(sgns[sort],
                                               splitting_points)}

    for dataset in datasets:
        data = snapshot.get_particles(dataset, part_type=part_type)[mask]
        grouped_data[dataset] = np.split(data[sort], splitting_points)

    return grouped_data


def sort_and_split_by_subhalo(snapshot, part_gns, part_sgns,
                              add_sort=None):
    """ Find indices that would sort particles first by subhalo and
    indices that would split the sorted array by subhalo.
    """

    if add_sort is None:
        add_sort = []

    # Sort, first by group number then by subgroup number, and finally
    # by potential additional sorting conditions:
    sort_arrs = tuple(add_sort + [part_sgns, part_gns])
    sort = np.lexsort(sort_arrs)

    subhalo_gns = snapshot.get_subhalos("GroupNumber")
    subhalo_sgns = snapshot.get_subhalos("SubGroupNumber")

    # Count number of entries with each group number and subgroup number
    # pairs:
    counts = [np.sum(np.logical_and(part_gns == gn, part_sgns == sgn))
              for gn, sgn in zip(subhalo_gns, subhalo_sgns)]

    splitting_points = np.cumsum(counts)[:-1]

    return sort, splitting_points


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
