import numpy as np
import math
import heapq


def get_data_for_matching(snap_ref, snap_exp):
    """ Retrieve datasets for matching for the given set of group numbers.

    Parameters
    ----------
    snap_ref : Snapshot object
        Reference snapshot.
    snap_exp : Snapshot object
        Explored snapshot.

    Returns
    -------
    (reference,explore) : tuple of dict
        Matching data for both snapshots in dictionaries.
    """

    reference = {'GNs': snap_ref.get_subhalos('GroupNumber'),
                 'SGNs': snap_ref.get_subhalos('SubGroupNumber'),
                 'IDs': snap_ref.get_subhalos_IDs(part_type=1),
                 'Mass': snap_ref.get_subhalos('MassType')[:, 1]}

    explore = {'GNs': snap_exp.get_subhalos('GroupNumber'),
               'SGNs': snap_exp.get_subhalos('SubGroupNumber'),
               'IDs': snap_exp.get_subhalos_IDs(part_type=1),
               'Mass': snap_exp.get_subhalos('MassType')[:, 1]}

    return reference, explore


def identify_group_numbers(gns1, gns2):
    """ Identifies the indices, where (gn,sgn) pairs align, between
    datasets 1 and 2.

    Parameters
    ----------
    gns1 : ndarray of int
        Group numbers of first dataset.
    gns2 : ndarray of int
        Group numbers of second dataset.

    Returns
    -------
    idx_of1_in2 : list of int
        Elements satisfy: GNs1[idx] == GNs2[idx_of1_in2[idx]] (unless
        GNs1[idx] not in GNs2)

    Notes
    -----
    If a certain pair in 1 does not exist in 2, it is identified with the
    halo with the same gn, for which sgn is the largest. """

    gns1 = gns1.astype(int)
    gns2 = gns2.astype(int)
    gn_cnt1 = np.bincount(gns1)
    gn_cnt2 = np.bincount(gns2)

    idx_of1_in2 = [None] * gns1.size
    for gn in gns1:
        for sgn in range(gn_cnt1[gn]):
            idx1 = np.sum(gn_cnt1[:gn]) + sgn
            # If halo with gn and sgn exists dataset 2, then identify
            # those halos. If not, set indices equal:
            if gn < gn_cnt2.size:
                if sgn < gn_cnt2[gn]:
                    idx2 = np.sum(gn_cnt2[:gn]) + sgn
            else:
                idx2 = min(gns2.size - 1, idx1)
            idx_of1_in2[idx1] = idx2

    return idx_of1_in2


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


class SubhaloMatcher:

    def __init__(self, no_match=2 ** 32, n_link_ref=15,
                 f_link_exp=1 / 5, f_mass_link=3):
        """

        Parameters
        ----------
        no_match : float
            Value used to indicate that no match is found.
        n_link_ref : int
            number of most bound in reference for matching
        f_link_exp : float
            Fraction of most bound in explored for matching
        f_mass_link : float
            Limit for mass difference between matching halos
        """

        self.f_link_exp = f_link_exp
        self.f_mass_link = f_mass_link
        self.n_link_ref = n_link_ref
        self.no_match = no_match

    def match_snapshots(self, snap_ref, snap_exp):
        """ Try matching all halos in snap_ref with given group numbers with
        halos in snap_exp with the same set of group numbers.

        Parameters
        ----------
        snap_ref : Snapshot object
            Reference snapshot.
        snap_exp : Snapshot object
            Explored snapshot.

        Returns
        -------
        matches_ref : ndarray of int of shape (# of halos,2)
            Group numbers in the first column and subgroup numbers in the
            second of matched halos in snap_exp. If no match was found for
            the halo in index i, then matches_ref[i] = [-1,-1].
        """

        reference, explore = get_data_for_matching(snap_ref,
                                                   snap_exp)

        # Initialize matches (ADD TO THE DICTIONARIES IN THE METHOD
        # ABOVE):
        matches_ref = self.no_match * np.ones(reference['GNs'].size,
                                              dtype=int)
        matches_exp = self.no_match * np.ones(explore['GNs'].size,
                                              dtype=int)
        out = -1 * np.ones((reference['GNs'].size, 2), dtype=int)

        # This is defined just make the code run a bit faster
        init_idents = identify_group_numbers(reference['GNs'],
                                             explore['GNs'])

        # Initialize priority queue:
        pq = []
        for idx_ref, idx_exp in enumerate(init_idents):
            heapq.heappush(pq, (0, (idx_ref, idx_exp)))

        trials = 0
        n_matched = 0
        while len(pq) > 0:
            trials += 1

            # Get next one for matching:
            next_item = heapq.heappop(pq)
            step = next_item[0]
            idx_ref = next_item[1][0]
            idx_exp0 = next_item[1][1]

            # Get index of the halo to be tried next. step tells how far
            # to iterate from initial index:
            idx_exp = get_index_at_step(idx_exp0, step,
                                        explore['GNs'].size)

            # Match:
            found_match = self.is_a_match(explore['IDs'][idx_exp],
                                          explore['Mass'][idx_exp],
                                          reference['IDs'][idx_ref],
                                          reference['Mass'][idx_ref])

            if found_match:
                n_matched += 1
                matches_ref[idx_ref] = idx_exp
                matches_exp[idx_exp] = idx_ref
                out[idx_ref] = [explore['GNs'][idx_exp],
                                explore['SGNs'][idx_exp]]
            else:
                new_step = self.iterate_step(idx_exp0, step, matches_exp)
                # If new_step == step, then all potential matches_ref for
                # idx_ref have been explored:
                if new_step != step:
                    heapq.heappush(pq, (new_step, (idx_ref, idx_exp0)))

        print("{} -> {}: {} trials, {} matches".format(snap_ref.snap_id,
                                                       snap_exp.snap_id,
                                                       trials, n_matched))
        return out

    def iterate_step(self, idx_ref, step_start, matches,
                     one_to_one=False):
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
        term = 10000

        step = step_start
        while step < term:

            idx = get_index_at_step(idx_ref, step + 1, matches.size)

            # If all values of array are consumed:
            if idx == idx_ref:
                break

            # If next index has not yet been matched:
            if matches[idx] == self.no_match or not one_to_one:
                step += 1
                break

            step += 1

        return step

    def is_a_match(self, ids_ref, mass_ref, ids_exp, mass_exp):
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

        found_match = False

        # Ignore pure gas halos:
        if mass_ref > 0 and mass_exp > 0:

            # Check masses:
            if (mass_ref / mass_exp < self.f_mass_link) and \
                    (mass_ref / mass_exp > 1 / self.f_mass_link):

                # Get most bound particles:
                most_bound_ref = ids_ref[:self.n_link_ref]
                most_bound_exp = ids_exp[:int(ids_exp.size \
                                              * self.f_link_exp)]

                # Check intersection:
                shared_parts = np.intersect1d(most_bound_ref,
                                              most_bound_exp,
                                              assume_unique=True)
                if len(shared_parts) > self.n_link_ref / 2:
                    found_match = True

        return found_match

# def find_match(subhalo, snap_id, snap_exp):
#    """ Attempts to match a given subhalo with another subhalo in a given
#    snapshot.
#
#    Parameters
#    ----------
#    subhalo : Subhalo object
#        Matched subhalo.
#    snap_id : int
#        ID of the snapshot, at which the match is searched.
#    snap_exp : Snapshot object
#        Explored snapshot.
#
#    Returns
#    -------
#    match : tuple
#        (gn,sgn) of matching halo in snap. match==(-1,-1) if no match is
#        found.
#    """
#
#    ids = subhalo.get_ids(snap_id)
#    mass = subhalo.get_halo_data('MassType', snap_id)[1]
#    gn, sgn = subhalo.tracer.get(snap_id)
#
#    # Set maximum number of iterations:
#    term = 10000
#
#    # Read subhalos with group numbers and subgroup numbers near gn and
#    # sgn:
#    fnums = neighborhood(snap_exp, gn, sgn, term / 2)
#    gns = snap_exp.get_subhalos('GroupNumber', fnums=fnums)
#    sgns = snap_exp.get_subhalos('SubGroupNumber', fnums=fnums)
#    ids_in_file = snap_exp.get_subhalos_IDs(part_type=1, fnums=fnums)
#    mass_in_file = snap_exp.get_subhalos('MassType', fnums=fnums)[:, 1]
#
#    # Get index of halo with same sgn and gn as ref:
#    idx0 = np.argwhere(np.logical_and((gns == gn), (sgns == sgn)))[0, 0]
#    #    print('find match for:', gns[idx0], sgns[idx0], ' in ',
#    #          snap_exp.snap_id)
#
#    # Initial value of match is returned if no match is found:
#    match = (-1, -1)
#
#    idx = idx0
#    for step in range(1, term):
#        ids_exp = ids_in_file[idx]
#        mass_exp = mass_in_file[idx]
#        found_match = is_a_match(ids, mass, ids_exp, mass_exp)
#
#        if found_match:
#            match = (gns[idx], sgns[idx])
#            break
#
#        idx = get_index_at_step(idx0, step, gns.size)
#        if idx == idx0:
#            break
#
#    return match
#
#
# def neighborhood(snap, gn, sgn, min_halos):
#    """ Gets file numbers of files that contain a minimum amount of
#    halos above and below a certain halo. """
#
#    fnum = snap.file_of_halo(gn, sgn)
#    GNs = snap.get_subhalos('GroupNumber', fnums=[fnum])
#    SGNs = snap.get_subhalos('SubGroupNumber', fnums=[fnum])
#    idx = np.argwhere(np.logical_and((GNs == gn), (SGNs == sgn)))[0, 0]
#
#    fnums = [fnum]
#    halos_below = idx
#    for n in range(fnum - 1, -1, -1):
#        if halos_below < min_halos:
#            fnums.append(n)
#            halos_below += snap.get_subhalos('GroupNumber', fnums=[n]).size
#        else:
#            break
#
#    with h5py.File(snap.grp_file, 'r') as grpf:
#        n_files = grpf['link1/FOF'].attrs.get('NTask')
#
#    halos_above = GNs.size - 1 - idx
#    for n in range(fnum + 1, n_files):
#        if halos_above < min_halos:
#            fnums.append(n)
#            halos_above += snap.get_subhalos('GroupNumber', fnums=[n]).size
#        else:
#            break
#
#    return fnums
