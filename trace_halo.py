import numpy as np
import h5py
import math
import heapq

from snapshot_obj import Snapshot


def trace_all(snap_init, gns=[], stop=101):
    """ Traces all subhalos of given galaxies as far back in time as 
    possible, starting from the given snapshot.

    Parameters
    ----------
    snap_init : Dataset object
        Starting point for the tracing.
    gns : int, optional
        Group numbers of the traced halos in the initial snapshot.
    stop : int, optional
        Earliest snapshot to be explored

    Returns
    -------
    tracer : dict of tuple
        Dictionary tracing the gn and sgn values of the halo through 
        snapshots. The keys are the snapshot IDs. The corresponding 
        redshifts are included as the first element of the tuples (the
        following elements being the gn and the sgn).
    """

    with h5py.File(snap_init.grp_file, 'r') as grpf:
        z_init = grpf['link0/Header'].attrs.get('Redshift')

    # Initialize tracer:
    tracer = {snap_init.snap_id: (z_init, gn, sgn)}
    snap = snap_init

    if direction == 'forward':
        condition = lambda ID: ID < 127
        add = 1
    else:
        condition = lambda ID: ID > stop
        add = -1

    while condition(snap.snap_id):
        snap_next = Snapshot(snap.sim_id, snap.snap_id + add)
        gn, sgn = find_match(snap_next, snap, gn, sgn)

        # No matching halo found:
        if gn == -1: break

        with h5py.File(snap_next.grp_file, 'r') as grpf:
            z_next = grpf['link0/Header'].attrs.get('Redshift')

        # Add match to tracer:
        tracer[snap_next.snap_id] = (z_next, gn, sgn)

        snap = snap_next

    return tracer


def match_all(snap_ref, snap_exp, gns=[]):
    """ Try matching all halos in snap_ref with given group numbers with
    halos in snap_exp with the same set of group numbers. 
    
    Parameters
    ----------
    snap_ref : Snapshot object
        Reference snapshot.
    snap_exp : Snapshot object
        Explored snapshot.
    gns : list of int, optional
        Group numbers that are to be matched. If empty, match all.

    Returns
    -------
    matches_ref : ndarray of int of shape (# of halos,2)
        Group numbers in the first column and subgroup numbers in the
        second of matched halos in snap_exp. If no match was found for
        the halo in index i, then matches_ref[i] = [-1,-1].
    """

    reference, explore = get_data_for_matching(snap_ref, snap_exp, gns)

    # Initialize match arrays:
    matches_exp = -1 * np.ones(2 * explore['GNs'].size) \
        .reshape((explore['GNs'].size, 2))
    matches_ref = -1 * np.ones(2 * reference['GNs'].size) \
        .reshape((reference['GNs'].size, 2))

    init_idents = identify_groupNumbers(reference['GNs'], explore['GNs'])

    # Initialize priority queue:
    pq = []
    for idx_ref, idx_exp in enumerate(init_idents):
        heapq.heappush(pq, (0, (idx_ref, idx_exp)))

    while len(pq) > 0:

        # Get next one for matching:
        next_item = heapq.heappop(pq)
        step = next_item[0]
        idx_ref = next_item[1][0]
        idx_exp0 = next_item[1][1]

        # Get index of the halo to be tried next. step tells how far to
        # iterate from initial index:
        idx_exp = get_index_at_step(idx_exp0, step, explore['GNs'].size)

        # Match:
        found_match = is_a_match(explore['IDs'][idx_exp],
                                 explore['Mass'][idx_exp],
                                 reference['IDs'][idx_ref],
                                 reference['Mass'][idx_ref])

        if found_match:
            matches_exp[idx_exp] = (reference['GNs'][idx_ref],
                                    reference['SGNs'][idx_ref])
            matches_ref[idx_ref] = (explore['GNs'][idx_exp],
                                    explore['SGNs'][idx_exp])
        else:
            new_step = iterate_step(idx_exp0, step, matches_exp)
            # If new_step == step, then all potential matches for idx_ref
            # have been explored:
            if new_step != step:
                heapq.heappush(pq, (new_step, (idx_ref, idx_exp0)))

    return matches_ref


def get_data_for_matching(snap_ref, snap_exp, gns):
    """ Retrieve datasets for matching for the given set of groupnumbers.
    
    Parameters
    ----------
    snap_ref : Snapshot object
        Reference snapshot.
    snap_exp : Snapshot object
        Explored snapshot.
    gns : list of int
        Group numbers that are to be matched. If empty, match all.

    Returns
    -------
    (reference,explore) : tuple of dict
        Matching data for both snapshots in dictionaries.
    """

    GNs_ref = snap_ref.get_subhalos('GroupNumber')
    GNs_exp = snap_exp.get_subhalos('GroupNumber')

    mask_ref = [True] * GNs_ref.size
    mask_exp = [True] * GNs_exp.size
    if gns:
        mask_ref = [gn in gns for gn in GNs_ref]
        mask_exp = [gn in gns for gn in GNs_exp]

    reference = {}
    reference['GNs'] = GNs_ref[mask_ref]
    reference['SGNs'] = snap_ref.get_subhalos('SubGroupNumber')[mask_ref]
    reference['IDs'] = snap_ref.get_subhalos_IDs(part_type=1)[mask_ref]
    reference['Mass'] = snap_ref.get_subhalos('Mass')[mask_ref]

    explore = {}
    explore['GNs'] = GNs_exp[mask_exp]
    explore['SGNs'] = snap_exp.get_subhalos('SubGroupNumber')[mask_exp]
    explore['IDs'] = snap_exp.get_subhalos_IDs(part_type=1)[mask_exp]
    explore['Mass'] = snap_exp.get_subhalos('Mass')[mask_exp]

    return reference, explore


def identify_groupNumbers(GNs1, GNs2):
    """ Identifies the indeces, where (gn,sgn) pairs align, between 
    datasets 1 and 2.

    Parameters
    ----------
    GNs1 : ndarray of int
        Group numbers of first dataset.
    GNs2 : ndarray of int
        Group numbers of second dataset.

    Returns
    -------
    idxOf1In2 : list of int
        Elements satisfy: GNs1[idx] == GNs2[idxOf1In2[idx]] (unless
        GNs1[idx] not in GNs2)
    
    Notes
    -----
    If a certain pair in 1 does not exist in 2, it is identified with the
    halo with the same gn, for which sgn is the largest. """

    GNs1 = GNs1.astype(int)
    GNs2 = GNs2.astype(int)
    gn_cnt1 = np.bincount(GNs1)
    gn_cnt2 = np.bincount(GNs2)

    idxOf1In2 = [None] * GNs1.size
    for gn in GNs1:
        for sgn in range(gn_cnt1[gn]):
            idx1 = np.sum(gn_cnt1[:gn]) + sgn
            # If halo with gn and sgn exists dataset 2, then identify
            # those halos. If not, set indeces equal:
            if gn < gn_cnt2.size:
                if sgn < gn_cnt2[gn]:
                    idx2 = np.sum(gn_cnt2[:gn]) + sgn
            else:
                idx2 = min(GNs2.size - 1, idx1)
            idxOf1In2[idx1] = idx2

    return idxOf1In2


def iterate_step(idx_ref, step_start, matches, oneToOne=False):
    """ Find the next index, which is nearest to idx_ref and has not yet
    been matched, for matching.

    Parameters
    ----------
    idx_ref : int
        Starting point of iteration.
    step_start : int
        Current step at function call.
    matches : ndarray 
        Array of already found matches of subhalos in reference snapshot.
    
    Returns
    -------
    step : int 
        The new step.

    Notes
    -----
        step is the number of steps it takes to iterate from idx_ref to 
        the next index.
    """

    # Set maximum number of iterations:
    term = 60

    step = step_start
    while step < term:

        idx = get_index_at_step(idx_ref, step + 1,
                                np.size(matches, axis=0))

        # If all values of array are consumed:
        if idx == idx_ref:
            break

        # If next index has not yet been matched:
        if matches[idx, 0] == -1 or not oneToOne:
            step += 1
            break

        step += 1

    return step


def get_index_at_step(idx_ref, step, lim):
    """ Get the index of the next subhalo after step iterations from
    idx_ref. 
    
    Parameters
    ----------
    idx_ref : int
        Starting point of iterations.
    step : int
        Number of iterations completed.
    lim : int
        Upper limit for index value.

    Returns
    -------
    idx : int
        Index of next subhalo after step iterations from idx_ref.
    """

    # Iterate outwards from idx_ref, alternating between lower and higher
    # index:
    idx = idx_ref + int(math.copysign( \
        math.floor((step + 1) / 2), (step % 2) - 0.5))

    # Check that index is not negative:
    if abs(idx_ref - idx) > idx_ref:
        idx = step
    # Check that index is not out of array bounds:
    elif abs(idx_ref - idx) > lim - 1 - idx_ref:
        idx = lim - step

    # If all values of array are consumed:
    if idx < 0 or idx >= lim:
        idx = idx_ref

    return idx


def find_match(subhalo, snap_id, snap_exp):
    """ Attempts to match a given subhalo with another subhalo in a given
    snapshot.

    Parameters
    ----------
    subhalo : Subhalo object
        Matched subhalo.
    snap_id : int
        ID of the snapshot, at which the match is searched.
    snap_exp : Snapshot object
        Explored snapshot.

    Returns
    -------
    match : tuple
        (gn,sgn) of matching halo in snap. match==(-1,-1) if no match is
        found.
    """

    ids = subhalo.get_ids(snap_id)
    mass = subhalo.get_halo_data('Mass', snap_id)
    gn, sgn = subhalo.tracer.get(snap_id)

    # Set maximum number of iterations:
    term = 100

    # Read subhalos with group numbers and subgroup numbers near gn and
    # sgn:
    fnums = neighborhood(snap_exp, gn, sgn, term / 2)
    gns = snap_exp.get_subhalos('GroupNumber', fnums=fnums)
    sgns = snap_exp.get_subhalos('SubGroupNumber', fnums=fnums)
    ids_in_file = snap_exp.get_subhalos_IDs(part_type=1, fnums=fnums)
    mass_in_file = snap_exp.get_subhalos('Mass', fnums=fnums)

    # Get index of halo with same sgn and gn as ref:
    idx0 = np.argwhere(np.logical_and((gns == gn), (sgns == sgn)))[0, 0]
#    print('find match for:', gns[idx0], sgns[idx0], ' in ',
#          snap_exp.snap_id)

    # Initial value of match is returned if no match is found:
    match = (-1, -1)

    idx = idx0
    for step in range(1, term):
#        print('idx=', idx)
        ids_exp = ids_in_file[idx]
        mass_exp = mass_in_file[idx]
        found_match = is_a_match(ids, mass, ids_exp, mass_exp)

        if found_match:
            match = (gns[idx], sgns[idx])
            break

        idx = get_index_at_step(idx0, step, gns.size)
        if idx == idx0:
            break

    return match


def neighborhood(snap, gn, sgn, min_halos):
    """ Gets file numbers of files that contain a minimum amount of
    halos above and below a certain halo. """

    fnum = snap.file_of_halo(gn, sgn)
    GNs = snap.get_subhalos('GroupNumber', fnums=[fnum])
    SGNs = snap.get_subhalos('SubGroupNumber', fnums=[fnum])
    idx = np.argwhere(np.logical_and((GNs == gn), (SGNs == sgn)))[0, 0]

    fnums = [fnum]
    halos_below = idx
    for n in range(fnum - 1, -1, -1):
        if halos_below < min_halos:
            fnums.append(n)
            halos_below += snap.get_subhalos('GroupNumber', fnums=[n]).size
        else:
            break

    with h5py.File(snap.grp_file, 'r') as grpf:
        n_files = grpf['link1/FOF'].attrs.get('NTask')

    halos_above = GNs.size - 1 - idx
    for n in range(fnum + 1, n_files):
        if halos_above < min_halos:
            fnums.append(n)
            halos_above += snap.get_subhalos('GroupNumber', fnums=[n]).size
        else:
            break

    return fnums


def is_a_match(IDs_ref, mass_ref, IDs_exp, mass_exp,
               restrict_exp=True):
    """ Check if two halos with given IDs and masses correspond to the same
    halo. 

    Parameters
    ----------
    IDs_ref : ndarray of int
        Particle IDs of the reference halo in ascending order by binding
        energy (most bound first).
    mass_ref : float
        Mass of (particles of) first halo.
    IDs_exp : ndarray of int
        Particle IDs of the explored halo in ascending order by binding
        energy (most bound first).
    mass_exp : float
        Mass of (particles of) second halo.

    Returns
    -------
    found_match : bool
        True iff halos match.

    Notes
    -----
    The reference subhalo can only be matched with one subhalo in the
    explored dataset. Explicitly, this is because of the following: if at
    least half of the N_link most bound particles in the reference subhalo
    are found in a given halo S, then there can be no other subhalo
    satisfying the same condition (no duplicates of any ID in dataset).
    """

    N_link = 20  # number of most bound in reference for matching
    f_exp = 1 / 5  # number of most bound in explored for matching
    frac_mass = 3  # Limit for mass difference between matching halos

    mostBound_ref = IDs_ref[:N_link]
    if restrict_exp:
        mostBound_exp = IDs_exp[:int(IDs_exp.size * f_exp)]
    else:
        mostBound_exp = IDs_exp

    shared_parts = np.intersect1d(mostBound_ref, mostBound_exp, \
                                  assume_unique=True)
    found_match = False
    if (len(shared_parts) > N_link / 2) and \
            (mass_ref / mass_exp < frac_mass) and \
            (mass_ref / mass_exp > 1 / frac_mass):
        found_match = True

    return found_match

