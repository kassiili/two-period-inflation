import numpy as np
import h5py
import math
import heapq

from snapshot_obj import Snapshot
from iteratearray import IterateArray


def trace_all(snap_init, gns=[], stop=101):
    """ Traces all subhalos of given galaxies as far back in time as
    possible, starting from the given snapshot.

    Parameters
    ----------
    snap_init : Snapshot object
        Starting point for the tracing.
    gns : int, optional
        Group numbers of the traced halos in the initial snapshot.
    stop : int, optional
        Earliest snapshot to be explored

    Returns
    -------
    tracer : ndarray of lists
        The lists trace the indices of the subhalos through snapshots.
        Each list element is a tuple, where the first entry is the
        snap_id and the second entry is the idx of the subhalo in that
        snapshot.
    """

    tracer = [[(snap_init.snap_id, i)] for i in
              range(snap_init.get_halo_number(gns))]
    snap = snap_init
    while snap.snap_id > stop:
        snap_next = Snapshot(snap.sim_id, snap.snap_id - 1)
        matches = match_all(snap, snap_next, gns)
        for halo_tracer in tracer:
            last_idx = halo_tracer[-1][1]
            if not last_idx is None:
                halo_tracer += [(snap_next.snap_id, matches[last_idx])]

        snap = snap_next

    tracer = np.array(tracer)
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

    # Initialize matches:
    matches_ref = np.array([None] * reference['GNs'].size)
    matches_exp = np.array([None] * explore['GNs'].size)  # This is
    # defined just make the code run a bit faster

    init_idents = identify_group_numbers(reference['GNs'], explore['GNs'])

    # Initialize iterator:
    mass = snap_exp.get_subhalos("MassType")[:, 1]
    iterator = IterateArray(init_idents, mass, term=100)

    # Initialize priority queue:
    pq = []
    for idx_ref in range(init_idents.size):
        heapq.heappush(pq, (0, idx_ref))

    pmax = 0
    while len(pq) > 0:
        # Get next one for matching:
        p, idx_ref = heapq.heappop(pq)  # [1]
        if p > pmax:
            pmax = p

        # Get index of the halo to be tried next:
        idx_exp = iterator.iterate(idx_ref)
#        if idx_exp is None:
#            print(idx_exp, p)
#        elif reference['GNs'][idx_ref] == 1 and \
#                explore['GNs'][idx_exp] == 1:
#            print(p, idx_ref, idx_exp)

        # If there are still untried candidates:
        if idx_exp is not None:
            found_match = False
            # If current halo is not matched:
            if matches_exp[idx_exp] is None:
                # Match:
                found_match = is_a_match(explore['IDs'][idx_exp],
                                         explore['Mass'][idx_exp],
                                         reference['IDs'][idx_ref],
                                         reference['Mass'][idx_ref])

            if found_match:
                matches_ref[idx_ref] = idx_exp
                matches_exp[idx_exp] = idx_ref
            else:
                # Set priority equal to the number previous matching
                # events:
                priority = iterator.get_step(idx_ref)
                heapq.heappush(pq, (priority, idx_ref))

    print(pmax)
    return matches_ref


def get_data_for_matching(snap_ref, snap_exp, gns):
    """ Retrieve datasets for matching for the given set of group numbers.

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

    gns_ref = snap_ref.get_subhalos('GroupNumber')
    gns_exp = snap_exp.get_subhalos('GroupNumber')

    mask_ref = [True] * gns_ref.size
    mask_exp = [True] * gns_exp.size
    if gns:
        mask_ref = [gn in gns for gn in gns_ref]
        #mask_exp = [gn in gns for gn in gns_exp]

    reference = {'GNs': gns_ref[mask_ref],
                 'SGNs': snap_ref.get_subhalos('SubGroupNumber')[mask_ref],
                 'IDs': snap_ref.get_subhalos_IDs(part_type=1)[mask_ref],
                 'Mass': snap_ref.get_subhalos('MassType')[:, 1][mask_ref]}

    explore = {'GNs': gns_exp[mask_exp],
               'SGNs': snap_exp.get_subhalos('SubGroupNumber')[mask_exp],
               'IDs': snap_exp.get_subhalos_IDs(part_type=1)[mask_exp],
               'Mass': snap_exp.get_subhalos('MassType')[:, 1][mask_exp]}

    return reference, explore


def identify_group_numbers(gns1, gns2):
    """ Identifies the indices, where (gn,sgn) pairs align, between
    snapshots 1 and 2.

    Parameters
    ----------
    gns1 : ndarray of int
        Group numbers of first dataset.
    gns2 : ndarray of int
        Group numbers of second dataset.

    Returns
    -------
    idx_of1_in2 : ndarray of int
        Where a group number and subgroup number pair exists in bots
        datasets, the indices are identified, i.e.:
            gns1[idx] == gns2[idx_of1_in2[idx]]
            (and the same for subgroup numbers)
        If a certain pair in 1 does not exist in 2, it is identified with
        the halo with the same gn, for which sgn is the largest.

    Notes
    -----
    There could be elemnts in gns2 that are not identified with any
    element in gns1.
    """

    gns1 = gns1.astype(int)
    gns2 = gns2.astype(int)
    gn_cnt1 = np.bincount(gns1)
    gn_cnt2 = np.bincount(gns2)

    idx_of1_in2 = np.zeros(gns1.size)
    for gn, cnt in enumerate(gn_cnt1):
        prev1 = np.sum(gn_cnt1[:gn])
        prev2 = np.sum(gn_cnt2[:gn])
        for sgn in range(cnt):
            idx1 = prev1 + sgn
            # If halo with gn and sgn exists snapshot 2, then identify
            # those halos. If not, set indices equal:
            if gn < gn_cnt2.size:
                if sgn < gn_cnt2[gn]:
                    idx2 = prev2 + sgn
            else:
                idx2 = min(gns2.size - 1, idx1)
            idx_of1_in2[idx1] = idx2

    return idx_of1_in2


def iterate_step(idx_ref, step_start, matches, one_to_one=False):
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
    one_to_one : bool, optional
        ???

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

        idx = get_index_at_step(idx_ref, step + 1, matches.size)

        # If all values of array are consumed:
        if idx == idx_ref:
            break

        # If next index has not yet been matched:
        if matches[idx] is None or not one_to_one:
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
    idx = idx_ref + int(math.copysign(
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
    mass = subhalo.get_halo_data('MassType', snap_id)[1]
    gn, sgn = subhalo.tracer.get(snap_id)

    # Set maximum number of iterations:
    term = 100

    # Read subhalos with group numbers and subgroup numbers near gn and
    # sgn:
    fnums = neighborhood(snap_exp, gn, sgn, term / 2)
    gns = snap_exp.get_subhalos('GroupNumber', fnums=fnums)
    sgns = snap_exp.get_subhalos('SubGroupNumber', fnums=fnums)
    ids_in_file = snap_exp.get_subhalos_IDs(part_type=1, fnums=fnums)
    mass_in_file = snap_exp.get_subhalos('MassType', fnums=fnums)[:, 1]

    # Get index of halo with same sgn and gn as ref:
    idx0 = np.argwhere(np.logical_and((gns == gn), (sgns == sgn)))[0, 0]
    #    print('find match for:', gns[idx0], sgns[idx0], ' in ',
    #          snap_exp.snap_id)

    # Initial value of match is returned if no match is found:
    match = (-1, -1)

    idx = idx0
    for step in range(1, term):
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


def is_a_match(ids_ref, mass_ref, ids_exp, mass_exp,
               limit_ids_exp=True):
    """ Check if two halos with given IDs and masses correspond to the same
    halo. 

    Parameters
    ----------
    ids_ref : ndarray of int
        Particle IDs of the reference halo in ascending order by binding
        energy (most bound first).
    mass_ref : float
        Mass of (particles of) first halo.
    ids_exp : ndarray of int
        Particle IDs of the explored halo in ascending order by binding
        energy (most bound first).
    mass_exp : float
        Mass of (particles of) second halo.
    limit_ids_exp : bool, optional
        Decides whether to limit the number most bound particles of the
        explored snapshot that are considered.

    Returns
    -------
    found_match : bool
        True iff halos match.

    Notes
    -----
    The reference subhalo can only be matched with one subhalo in the
    explored dataset. Explicitly, this is because of the following: if at
    least half of the n_link most bound particles in the reference subhalo
    are found in a given halo S, then there can be no other subhalo
    satisfying the same condition (no duplicates of any ID in dataset).
    """

    n_link = 20  # number of most bound in reference for matching
    f_exp = 1 / 5  # number of most bound in explored for matching
    frac_mass = 3  # Limit for mass difference between matching halos

    most_bound_ref = ids_ref[:n_link]
    if limit_ids_exp:
        most_bound_exp = ids_exp[:int(ids_exp.size * f_exp)]
    else:
        most_bound_exp = ids_exp

    shared_parts = np.intersect1d(most_bound_ref, most_bound_exp,
                                  assume_unique=True)
    found_match = False
    if (len(shared_parts) > n_link / 2) and \
            (mass_ref / mass_exp < frac_mass) and \
            (mass_ref / mass_exp > 1 / frac_mass):
        found_match = True

    return found_match
