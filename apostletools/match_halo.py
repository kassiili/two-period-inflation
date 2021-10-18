import numpy as np
import math
import heapq


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


class SnapshotMatcher:

    def __init__(self, no_match=2 ** 32, n_link_ref=15,
                 f_link_exp=1 / 5, f_mass_link=3):
        """

        Parameters
        ----------
        no_match : int
            Value used to indicate that no match is found.
        n_link_ref : int
            Number of most bound in reference used for matching.
        f_link_exp : float
            Fraction of most bound in explored used for matching.
        f_mass_link : float
            Limit for mass difference between matching halos.
        """

        self.f_link_exp = f_link_exp
        self.f_mass_link = f_mass_link
        self.n_link_ref = n_link_ref
        self.no_match = no_match
        self.snap = None
        self.snap_search = None
        self.subhalo_num_search = -1

    def match_snapshots(self, snap, snap_search, max_num_matched=3):
        """ Try matching all subhalos in snap with subhalos in
        snap_search.

        Parameters
        ----------
        snap : Snapshot object
            The snapshot, whose subhalos are being matched.
        snap_search : Snapshot object
            The snapshot, whose subhalos are being tried as matches for
            subhalos in snap.
        max_num_matched : int, optional
            Maximum number of subhalos in self.snap, with which a single
            subhalo in snap_search can be matched.

        Returns
        -------
        matches : ndarray of int
            Array of indices of matched subhalos in snap_search.

        Notes
        -----
        A subhalo in snap is only matched with one subhalo in
        snap_search, but the same is not true of subhalos in snap_search.
        I.e. the identification from ref to exp can be described as  a
        function that is not necessarily injective. This feature is
        logically inherited from the is_a_match method.
        """

        self.snap = snap
        self.snap_search = snap_search
        self.subhalo_num_search = self.snap_search.get_subhalo_number()
        reference, explore = self.get_data_for_matching()

        # Initialize matches (ADD TO THE DICTIONARIES IN THE METHOD
        # ABOVE):
        matches = self.no_match * np.ones(reference['GNs'].size,
                                          dtype=int)
        matches_search = self.no_match * np.ones(
            (explore['GNs'].size, max_num_matched), dtype=int)

        # This is defined just make the code run a bit faster
        init_idents = identify_group_numbers(reference['GNs'],
                                             explore['GNs'])

        # Initialize priority queue:
        pq = []
        for idx, idx_search in enumerate(init_idents):
            heapq.heappush(pq, (0, (idx, idx_search)))

        trials = 0
        n_matched = 0
        while len(pq) > 0:
            trials += 1

            # Get next one for matching:
            next_item = heapq.heappop(pq)
            step = next_item[0]
            idx = next_item[1][0]  # Index of subhalo in snap
            idx_search_ref = next_item[1][1]  # Index of the
            # iteration reference point in snap_search

            # Get index of the halo to be tried next. step tells how far
            # to iterate from initial index:
            idx_search = self.get_index_at_step(idx_search_ref, step)

            # If all subhalos have been tried already:
            if idx_search == idx_search_ref and step > 0:
                continue

            # Match:
            # ARE THE ARGUMENTS THE WRONG WAY AROUND???
            found_match = self.is_a_match(explore['IDs'][idx_search],
                                          explore['Mass'][idx_search],
                                          reference['IDs'][idx],
                                          reference['Mass'][idx])

            if found_match:
                n_matched += 1
                matches[idx] = idx_search
                matches_search[idx_search] = self.add_to_matches(
                    matches_search[idx_search], idx, reference['Mass'])
            else:
                heapq.heappush(pq, (step + 1, (idx, idx_search_ref)))

        print("{} -> {}: {} trials, {} matches".format(snap.snap_id,
                                                       snap_search.snap_id,
                                                       trials, n_matched))
        return matches, matches_search

    def get_data_for_matching(self):
        """ Retrieve datasets for matching for the given set of group numbers.

        Returns
        -------
        (reference,explore) : tuple of dict
            Matching data for both snapshots in dictionaries.
        """

        reference = {'GNs': self.snap.get_subhalos('GroupNumber'),
                     'SGNs': self.snap.get_subhalos('SubGroupNumber'),
                     'IDs': self.snap.get_subhalos_IDs(part_type=1),
                     'Mass': self.snap.get_subhalos('MassType')[:, 1]}

        explore = {'GNs': self.snap_search.get_subhalos('GroupNumber'),
                   'SGNs': self.snap_search.get_subhalos(
                       'SubGroupNumber'),
                   'IDs': self.snap_search.get_subhalos_IDs(part_type=1),
                   'Mass': self.snap_search.get_subhalos('MassType')[:,
                           1]}

        return reference, explore

    def add_to_matches(self, matches, new_match, masses):
        """ Try to insert the new match into the array of existing
        matches. """

        if matches[1] < self.no_match:
            print(matches)

        # Try insertion, and move the rest of the matches accordingly,
        # if the insertion is successful:
        insertion = new_match
        for i, idx in enumerate(matches):
            if idx == self.no_match:
                matches[i] = insertion
                break

            if masses[idx] < masses[insertion]:
                matches[i] = insertion
                insertion = idx

        return matches

    def get_index_at_step(self, idx_ref, step):
        """ Get the index of the next subhalo after step iterations from
        idx_ref.

        Parameters
        ----------
        idx_ref : int
            Starting point of iterations.
        step : int
            Number of iterations completed.

        Returns
        -------
        idx : int
            Index of next subhalo after step iterations from idx_ref.
        """

        # Get the upper limit for the value of the index:
        lim = self.subhalo_num_search

        # Iterate outwards from idx_ref, alternating between lower and
        # higher indices:
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

# def find_match(subhalo, snap_id, snap_search):
#    """ Attempts to match a given subhalo with another subhalo in a given
#    snapshot.
#
#    Parameters
#    ----------
#    subhalo : Subhalo object
#        Matched subhalo.
#    snap_id : int
#        ID of the snapshot, at which the match is searched.
#    snap_search : Snapshot object
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
#    fnums = neighborhood(snap_search, gn, sgn, term / 2)
#    gns = snap_search.get_subhalos('GroupNumber', fnums=fnums)
#    sgns = snap_search.get_subhalos('SubGroupNumber', fnums=fnums)
#    ids_in_file = snap_search.get_subhalos_IDs(part_type=1, fnums=fnums)
#    mass_in_file = snap_search.get_subhalos('MassType', fnums=fnums)[:, 1]
#
#    # Get index of halo with same sgn and gn as ref:
#    idx0 = np.argwhere(np.logical_and((gns == gn), (sgns == sgn)))[0, 0]
#    #    print('find match for:', gns[idx0], sgns[idx0], ' in ',
#    #          snap_search.snap_id)
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
