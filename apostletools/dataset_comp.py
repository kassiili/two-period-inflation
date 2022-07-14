import numpy as np

from astropy import units
from astropy import constants


def split_luminous(snap):
    sm = snap.get_subhalos('Stars/Mass')
    mask_lum = (sm > 0)
    mask_dark = (sm == 0)
    return mask_lum, mask_dark


def prune_vmax(snap, low_lim=10 ** (-10), up_lim=10 ** 10):
    maxpoint = snap.get_subhalos("Max_Vcirc", h5_group="Extended")
    vmax = maxpoint[:, 0]
    low_lim = low_lim * units.km.to(units.cm)
    up_lim = up_lim * units.km.to(units.cm)
    mask = np.logical_and(vmax >= low_lim, vmax < up_lim)
    return mask


def split_satellites_by_distance(snap, m31_id, mw_id, sat_r=300, isol_r=2000,
                                 comov=True, exclude_sat_of_isol=True):
    """ Select satellites and isolated galaxies from subhalos by their
    distance to the M31 and MW galaxies and to the LG centre, respectively.

    Parameters
    ----------
    snap : Snapshot object
    m31_id : tuple of two int
        Group number and subgroup number of the M31 halo.
    mw1_id : tuple of two int
        Group number and subgroup number of the MW halo.
    sat_r : float, optional
        Maximum (physical) distance for satellite galaxies in units of kpc.
        Default value is 300 kpc.
    isol_r : float, optional
        Maximum (physical) distance for isolated galaxies in units of kpc.
        Default value is 2000 kpc.
    comov : bool, optional
        Indicates if sat_r and isol_r are given in comoving or physical
        units. Default value is False.
    exclude_sat_of_isol : bool, optional
        If True, only the central SUBFIND subhalos of isolated FOF groups are
        considered as isolated subhalos. Default value is True.

    Returns
    -------
    mask_m31_sat, mask_mw_sat, mask_isol : ndarray of bool

    Notes
    -----
    Satellites of a central halo are defined as subhalos lying within the
    distance max_dist_sat from the central. In case a satellite is within
    this distance from multiple centrals, it is assigned as a satellite of the
    central that is closest. Isolated galaxies are, by default, defined as the
    subhalos that are not bound to another halo (i.e. have subgroup
    number == 0), and lie within the distance max_dist_isol from the centre of
    the two centrals but are not satellites.
    """

    if comov:
        sat_r = comoving_distance_to_physical(snap, sat_r)
        isol_r = comoving_distance_to_physical(snap, isol_r)

    # Convert to cgs (which is the data units):
    sat_r = sat_r * units.kpc.to(units.cm)
    isol_r = isol_r * units.kpc.to(units.cm)

    cops = snap.get_subhalos("CentreOfPotential")

    # Get the masking array for M31 satellites:
    m31_idx = snap.index_of_halo(m31_id[0], m31_id[1])
    dist_to_m31 = distance_to_point(snap, cops[m31_idx])
    mask_m31_sat = within_distance_range(
        dist_to_m31,
        0.01 * units.kpc.to(units.cm),  # Exclude the central itself
        sat_r
    )

    # Get the masking array for MW satellites:
    mw_idx = snap.index_of_halo(mw_id[0], mw_id[1])
    dist_to_mw = distance_to_point(snap, cops[mw_idx])
    mask_mw_sat = within_distance_range(
        dist_to_mw,
        0.01 * units.kpc.to(units.cm),  # Exclude the central itself
        sat_r
    )

    # In case there is an intersection, identify satellites with the central
    # that is closer to them:
    mask_closer_to_m31 = (dist_to_m31 <= dist_to_mw)
    mask_m31_sat = np.logical_and(mask_m31_sat, mask_closer_to_m31)
    mask_mw_sat = np.logical_and(mask_mw_sat,
                                 np.logical_not(mask_closer_to_m31))

    # Select all but M31 or MW:
    sub_idx = np.arange(snap.get_subhalo_number())
    mask_not_central = np.logical_and(sub_idx != m31_idx, sub_idx != mw_idx)

    # Select isolated galaxies:
    dist_to_lg = distance_to_point(snap, compute_LG_centre(snap, m31_id, mw_id))
    mask_lg_member = within_distance_range(dist_to_lg, 0, isol_r)
    mask_isol = np.logical_and.reduce([mask_lg_member,
                                       np.logical_not(mask_m31_sat),
                                       np.logical_not(mask_mw_sat),
                                       mask_not_central])

    if exclude_sat_of_isol:
        sgns = snap.get_subhalos('SubGroupNumber')
        mask_isol = np.logical_and(mask_isol, sgns == 0)

    return mask_m31_sat, mask_mw_sat, mask_isol


# IS THIS REDUNDANT?
def split_satellites_by_distance_obj(snap, m31, mw, sat_r=300, isol_r=2000,
                                     comov=False, exclude_sat_of_isol=True):
    """ Select satellites and isolated galaxies from subhalos by their
    distance to the M31 and MW galaxies and to the LG centre, respectively.

    Parameters
    ----------
    snap : Snapshot object
    m31 : Subhalo object
        The M31 subhalo.
    mw : Subhalo object
        The MW subhalo.
    sat_r : float, optional
        Maximum (physical) distance for satellite galaxies in units of kpc.
        Default value is 300 kpc.
    isol_r : float, optional
        Maximum (physical) distance for isolated galaxies in units of kpc.
        Default value is 2000 kpc.
    comov : bool, optional
        Indicates if sat_r and isol_r are given in comoving or physical
        units. Default value is False.
    exclude_sat_of_isol : bool, optional
        If True, only the central SUBFIND subhalos of isolated FOF groups are
        considered as isolated subhalos. Default value is True.

    Returns
    -------
    mask_m31_sat, mask_mw_sat, mask_isol : ndarray of bool

    Notes
    -----
    Satellites of a central halo are defined as subhalos lying within the
    distance max_dist_sat from the central. In case a satellite is within
    this distance from multiple centrals, it is assigned as a satellite of the
    central that is closest. Isolated galaxies are, by default, defined as the
    subhalos that are not bound to another halo (i.e. have subgroup
    number == 0), and lie within the distance max_dist_isol from the centre of
    the two centrals but are not satellites.
    """

    if comov:
        sat_r = comoving_distance_to_physical(snap, sat_r)
        isol_r = comoving_distance_to_physical(snap, isol_r)

    # Convert to cgs (which is the data units):
    sat_r = sat_r * units.kpc.to(units.cm)
    isol_r = isol_r * units.kpc.to(units.cm)

    # Get the masking array for M31 satellites:
    dist_to_m31 = distance_to_point(
        snap, m31.get_halo_data('CentreOfPotential', snap.snap_id)[0]
    )
    mask_m31_sat = within_distance_range(
        dist_to_m31,
        0.01 * units.kpc.to(units.cm),  # Exclude the central itself
        sat_r
    )

    # Get the masking array for MW satellites:
    dist_to_mw = distance_to_point(
        snap, mw.get_halo_data('CentreOfPotential', snap.snap_id)[0]
    )
    mask_mw_sat = within_distance_range(
        dist_to_mw,
        0.01 * units.kpc.to(units.cm),  # Exclude the central itself
        sat_r
    )

    # In case there is an intersection, identify satellites with the central
    # that is closer to them:
    mask_closer_to_m31 = (dist_to_m31 <= dist_to_mw)
    mask_m31_sat = np.logical_and(mask_m31_sat, mask_closer_to_m31)
    mask_mw_sat = np.logical_and(mask_mw_sat,
                                 np.logical_not(mask_closer_to_m31))

    # Select all but M31 or MW:
    sub_idx = np.arange(snap.get_subhalo_number())
    mask_not_central = np.logical_and(
        sub_idx != m31.get_index_at_snap(snap.snap_id),
        sub_idx != mw.get_index_at_snap(snap.snap_id)
    )

    # Select isolated galaxies:
    dist_to_lg = distance_to_point(
        snap, compute_centre_of_subhalos(snap, [m31, mw])
    )
    mask_lg_member = within_distance_range(dist_to_lg, 0, isol_r)
    mask_isol = np.logical_and.reduce([mask_lg_member,
                                       np.logical_not(mask_m31_sat),
                                       np.logical_not(mask_mw_sat),
                                       mask_not_central])

    if exclude_sat_of_isol:
        sgns = snap.get_subhalos('SubGroupNumber')
        mask_isol = np.logical_and(mask_isol, sgns == 0)

    return mask_m31_sat, mask_mw_sat, mask_isol


def satellites_of_halo_by_distance(snap, halo, max_dist=300, comov=False):
    """ Select satellites from subhalos by their distance to the given halo.

    Parameters
    ----------
    snap : Snapshot object
    halo : Subhalo object
    max_dist : float, optional
        Maximum (physical) distance for satellite galaxies in units of kpc.
        Default value is 300 kpc.
    max_dist_isol : float, optional
        Maximum (physical) distance for isolated galaxies in units of kpc.
        Default value is 2000 kpc.

    Returns
    -------
    mask_sat : ndarray of bool
    """

    if comov:
        max_dist = comoving_distance_to_physical(snap, max_dist)

    halo_cop = halo.get_halo_data('CentreOfPotential', snap.snap_id)[0]
    dist_to_halo = distance_to_point(snap, halo_cop)
    min_dist = 0.01 * units.kpc.to(units.cm)  # Exclude the central itself
    max_dist = max_dist * units.kpc.to(units.cm)
    mask_sat = within_distance_range(dist_to_halo, min_dist, max_dist)

    return mask_sat


def compute_centre_of_subhalos(snap, subhalos):
    """ Compute the mean of the centres of the given subhalos. """
    cops = np.array([halo.get_halo_data('CentreOfPotential',
                                        snap_ids=[snap.snap_id])
                     for halo in subhalos])
    cops = periodic_wrap(snap, cops[0], cops)

    centre = np.mean(cops, axis=0)
    return centre


def split_satellites_by_r200(snap, m31_ident, mw_ident,
                             r200='Group_R_TopHat200', max_dist_isol=2000):
    """ Select satellites and isolated galaxies within r200 of the M31 and MW
    galaxies and within max_dist_isol of the LG barycentre, respectively.

    Parameters
    ----------
    snap : Snapshot object
    m31_ident : tuple of two int
        Group number and subgroup number of the M31 halo.
    mw1_ident : tuple of two int
        Group number and subgroup number of the MW halo.
    r200 : str, optional
        Name of the dataset in the data catalogues (different definitions of
        r200 are listed).
    max_dist_isol : float, optional
        Maximum (physical) distance for isolated galaxies in units of kpc.
        Default value is 2000 kpc.
    """

    if round(snap.get_attribute('Redshift', 'Header'), 5) > 0:
        max_dist_isol = comoving_distance_to_physical(snap, max_dist_isol)

    r200_m31 = snap.get_subhalos(r200, 'FOF')[m31_ident[0]] \
               * units.cm.to(units.kpc)
    r200_mw = snap.get_subhalos(r200, 'FOF')[mw_ident[0]] \
              * units.cm.to(units.kpc)
    print(r200, r200_m31, r200_mw, max_dist_isol)

    return split_satellites_by_distance_old(snap, m31_ident, mw_ident,
                                            max_dist_sat=[r200_m31, r200_mw],
                                            max_dist_isol=max_dist_isol)


def comoving_distance_to_physical(snap, comov_d):
    a = snap.get_attribute('Time', 'Header')
    return a * comov_d


def split_satellites_by_distance_old(snap, m31_ident, mw_ident,
                                     max_dist_sat=300,
                                     max_dist_isol=2000):
    """ Select satellites and isolated galaxies from subhalos by their
    distance to the M31 and MW galaxies and to the LG barycentre,
    respectively.

    Parameters
    ----------
    snap : Snapshot object
    m31_ident : tuple of two int
        Group number and subgroup number of the M31 halo.
    mw1_ident : tuple of two int
        Group number and subgroup number of the MW halo.
    max_dist_sat : float, optional
        Maximum (physical) distance for satellite galaxies in units of kpc.
        Default value is 300 kpc.
    max_dist_isol : float, optional
        Maximum (physical) distance for isolated galaxies in units of kpc.
        Default value is 2000 kpc.

    Returns
    -------
    masks_sat, mask_isol : ndarray of bool

    Notes
    -----
    Satellites of a central halo are defined as subhalos lying within the
    distance max_dist_sat from the central. In case a satellite is within
    this distance from multiple centrals, it is assigned as a satellite of the
    central that is closest. Isolated galaxies are defined as the subhalos
    that are not bound to another halo (i.e. have subgroup number == 0),
    and lie within the distance max_dist_isol from the barycentre of the two
    centrals but are not satellites.
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
    min_dist = 0.01 * units.kpc.to(units.cm)  # Exclude the central itself
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
    print("Computing v_circ at {} kpc for {} at snapshot {}...".format(
        r, snapshot.sim_id, snapshot.snap_id))
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
    print("Computing subhalo rotation curves for {} at snapshot {}...".format(
        snapshot.sim_id, snapshot.snap_id))
    cmass, radii = compute_mass_accumulation(snapshot, part_type=part_type)

    # Compute running average:
    radii = [np.array(r[n_soft - 1::n_soft]) for r in radii]
    cmass = [np.array(cm[n_soft - 1::n_soft]) for cm in cmass]

    myG = constants.G.to(units.cm ** 3 * units.g ** -1 * units.s **
                         -2).value
    v_circ = [np.sqrt(cm * myG / r) for cm, r in zip(cmass, radii)]

    # Add zero:
    radii = [np.concatenate((np.array([0]), r)) for r in radii]
    v_circ = [np.concatenate((np.array([0]), v)) for v in v_circ]

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
    raw_cmass = snapshot.get_subhalos('MassAccum', h5_group='Extended')
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


# _____ Crap below

def get_subhalos_from_simulation_awful(subhalos, simulation, dset_name,
                                       h5_group='Subhalo', snap_ids=None):
    if not snap_ids:
        data_dict = simulation.get_subhalos_in_snapshots(
            simulation.get_snap_ids(), dset_name, h5_group=h5_group
        )
    else:
        data_dict = simulation.get_subhalos_in_snapshots(snap_ids, dset_name,
                                                         h5_group=h5_group)

    return None


def get_subhalos_at_fallin(subhalos, fallin_dict, data_dict):
    """ Read subhalo entries from data_dict at the time of (last) fall-in. """

    # For each subhalo, get its index and the snapshot ID, in its latest
    # snapshot:
    last_inds = map(lambda l: (l[0][-1], l[1][-1]),
                    [sub.get_indices() for sub in subhalos])

    fallin_times = np.array([fallin_dict[sid][idx] for idx, sid in last_inds])
    # print(fallin_times[:150])

    fallin_data = np.empty(subhalos.size)
    for i, (sub, snap_id) in enumerate(zip(subhalos, fallin_times)):
        if np.isnan(snap_id):
            fallin_data[i] = np.nan
        else:
            fallin_data[i] = data_dict[snap_id][sub.get_index_at_snap(snap_id)]

    # fallin_data = np.empty(subhalos.size)
    # for sid, dataset in data_dict.items():
    #     mask = (fallin_times == sid)
    #     print(sid, np.sum(mask))
    #     print(mask.size, dataset.size)
    #     print(dataset[mask])
    #     fallin_data[mask] = dataset[mask]

    return fallin_data, fallin_times


def data_at_fallin(fallin_snaps, fallin_inds, data):
    """ Read satellite data at fall-in time, and return as a dictionary arrays,
    of the same shape as in the source data catalogues, for each snapshot.

    Parameters
    ----------
    fallin_snaps : dict
        Key - snapshot ID, value - subhalo array (of equal shape as the
        source data, and with the same subhalo ordering) of snapshot IDs of
        the first snapshots, where the subhalo was a satellite.
    fallin_inds : dict
        Key - snapshot ID, value - subhalo array (of equal shape as the
        source data, and with the same subhalo ordering) of subhalo indices
        in their respective fall-in snapshots.
    data : dict
        Key - snapshot ID, value - subhalo dataset in the given snapshot.

    Returns
    -------
    fallin_data : dict
        Key - snapshot ID, value - subhalo data value, when it fell in (the
        dictionary organization is the same as for the inputs).
    """

    fallin_data = {}

    # Iterate over snapshots:
    for snap_id in fallin_snaps.keys():
        # Initialize fall-in dataset at snap_id with NaN (this will be the
        # end value for subhalos that are not satellites):
        fallin_data[snap_id] = np.full(fallin_snaps[snap_id].size, np.nan)

        # Iterate over fall-in snapshots of satellites at snap_id:
        for fallin_snap_id in np.unique(fallin_snaps[snap_id]):
            if np.isnan(fallin_snap_id): continue

            # Computing masking array for satellites that fell in at
            # fallin_snap_id (and get their index places in fallin_snap_id):
            mask = (fallin_snaps[snap_id] == fallin_snap_id)
            inds = fallin_inds[snap_id][mask].astype(int)

            # Store the fall-in data of these satellites:
            fallin_data[snap_id][mask] = data[fallin_snap_id][inds]

    return fallin_data


def index_at_fallin(sub_dict, fallin_snaps):
    fallin_inds = {}

    # Iterate over snapshots:
    for snap_id in sub_dict.keys():
        # Initialize fall-in dataset at snap_id with NaN (this will be the
        # end value for subhalos that are not satellites):
        fallin_inds[snap_id] = np.full(fallin_snaps[snap_id].size, np.nan)

        # Iterate over satellites, saving their indices at the fall-in
        # snapshots:
        mask_sat = ~np.isnan(fallin_snaps[snap_id])
        fallin_inds[snap_id][mask_sat] = [
            subhalo.get_index_at_snap(fallin_snap_id)
            for subhalo, fallin_snap_id in
            zip(sub_dict[snap_id][mask_sat], fallin_snaps[snap_id][mask_sat])
        ]

    return fallin_inds


def get_subhalos_from_simulation(subhalos, simulation, dset_name,
                                 h5_group='Subhalo'):
    data_dict = simulation.get_subhalos_in_snapshots(
        simulation.get_snap_ids(), dset_name, h5_group=h5_group
    )

    sub_data = []
    sub_snaps = []
    for sub in subhalos:
        data, snap_ids = subhalo_dataset_from_dict(sub, data_dict)
        sub_data.append(data)
        sub_snaps.append(snap_ids)

    return sub_data, sub_snaps


def subhalo_dataset_from_dict(subhalo, data):
    """ Read subhalo data entries from a data dictionary.

    Parameters
    ----------
    subhalo : Subhalo object
    data : dict
        Key - snapshot, value - dataset in snapshot

    Returns
    -------
    sub_data, snap_ids : ndarray
        Data entries for ´subhalo´ and the respective snapshot IDs.
    """
    inds, snap_ids = subhalo.get_indices()
    subdata = np.array([
        data[sid][i] for i, sid in zip(inds, snap_ids)
    ])
    return subdata, snap_ids


def select_by_vmax_at_fallin(fallin_times, vmax_dict, vmax_lim):
    # THIS IS CRAP

    masks = {}
    for sid, fit_arr in fallin_times.items():
        # Select satellites (i.e. entries with not nan as fallin time):
        mask_sat = ~np.isnan(fit_arr)
        mask = mask_sat

        # Out of the satellites
        mask[mask_sat] = (vmax_dict[sid][mask_sat] > vmax_lim)

        masks[sid] = mask

    return mask

