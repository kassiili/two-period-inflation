import numpy as np


class SnapshotMatcher:

    def __init__(self, no_match=2 ** 32, n_link_ref=20,
                 f_link_srch=1 / 5, f_mass_link=3, n_matches=1):
        """

        Parameters
        ----------
        no_match : int
            Value used to indicate that no match is found.
        n_link_ref : int
            Number of most bound in reference used for matching.
        f_link_srch : float
            Fraction of most bound in explored used for matching.
        f_mass_link : float
            Limit for mass difference between matching halos.
        n_matches : int, optional
            Maximum number of subhalos in self.snap, with which a single
            subhalo in snap_search can be matched.
        """

        self.no_match = no_match
        self.n_link_ref = n_link_ref
        self.f_link_srch = f_link_srch
        self.f_mass_link = f_mass_link
        self.n_matches = n_matches

    def match_snapshots(self, snap_ref, snap_srch):
        """ Try matching all subhalos in snap_ref with subhalos in
        snap_search.

        Attributes
        ----------
        snap_ref : Snapshot object
            The snapshot, whose subhalos are being matched.
        snap_srch : Snapshot object
            The snapshot, whose subhalos are being tried as matches for
            subhalos in snap_ref.

        Returns
        -------
        matches : ndarray of int
            Array of indices of matched subhalos in snap_search.

        Notes
        -----
        A subhalo in snap_ref is only matched with one subhalo in
        snap_srch, but the same is not true of subhalos in snap_srch.
        I.e. the identification from ref to exp can be described as  a
        function that is not necessarily injective. This feature is
        logically inherited from the is_a_match method.
        """

        if self.n_matches == 1:
            return self.match_injective(snap_ref, snap_srch)
        else:
            return self.match_non_injective(snap_ref, snap_srch)

    # NOT TESTED:
    def match_injective(self, snap_ref, snap_srch):
        """ Match injectively subhalos in snap_ref with subhalos in snap_search.

        Attributes
        ----------
        snap_ref : Snapshot object
            The snapshot, whose subhalos are being matched.
        snap_srch : Snapshot object
            The snapshot, whose subhalos are being tried as matches for
            subhalos in snap_ref.

        Returns
        -------
        matches_ref, matches_srch : ndarray of int
            Array of indices of matched subhalos in snap_search (snap_ref).
        """

        mass_ref = snap_ref.get_subhalos('MassType')[:, 1]
        ids_ref = snap_ref.get_subhalos_IDs(part_type=[1])
        mass_srch = snap_srch.get_subhalos('MassType')[:, 1]
        ids_srch = snap_srch.get_subhalos_IDs(part_type=[1])

        matches_ref = self.no_match * np.ones(mass_ref.size, dtype=int)
        matches_srch = self.no_match * np.ones(mass_srch.size, dtype=int)
        mask_unmatched_srch = np.full(mass_srch.size, True)

        # Iterate over subhalos in ref, in descending order by mass, s.t. if
        # two subhalos could be linked with the same subhalo in snap_srch,
        # only the more massive one is linked:
        sorting_ref = np.argsort(-mass_ref)
        for sub_idx_ref in sorting_ref:
            # Select match candidates from snap_search:
            mask_mass_range = self.select_by_mass(
                mass_ref[sub_idx_ref], mass_srch
            )

            # Look for matches among unmatched, in the given mass range:
            searched_subs = np.arange(mass_srch.size)[
                np.logical_and(mask_unmatched_srch, mask_mass_range)
            ]
            for sub_idx_srch in searched_subs:
                # Match:
                found_match = self.is_a_match(ids_ref[sub_idx_ref],
                                              mass_ref[sub_idx_ref],
                                              ids_srch[sub_idx_srch],
                                              mass_srch[sub_idx_srch])

                if found_match:
                    matches_ref[sub_idx_ref] = sub_idx_srch
                    matches_srch[sub_idx_srch] = sub_idx_ref
                    mask_unmatched_srch[sub_idx_srch] = False
                    break

        return matches_ref, matches_srch

    # # MADE THIS BY MISTAKE: the order, in which subhalos in snap_srch are
    # # iterated over, does not influence the linking output, by virtue of how
    # # ´is_a_match´ is implemented (there can only be one matching subhalo in
    # # snap_srch for each subhalo in snap_ref). That order does matter for the
    # # time complexity, however: most matches have similar GNs, and thus,
    # # by default the arrays are in a befitting order (and re-ordering by mass
    # # only slows things down)
    # def match_injective_reordered(self, snap_ref, snap_srch):
    #     """ Match injectively subhalos in snap_ref with subhalos in snap_search.
    #
    #     Attributes
    #     ----------
    #     snap_ref : Snapshot object
    #         The snapshot, whose subhalos are being matched.
    #     snap_srch : Snapshot object
    #         The snapshot, whose subhalos are being tried as matches for
    #         subhalos in snap_ref.
    #
    #     Returns
    #     -------
    #     matches_ref, matches_srch : ndarray of int
    #         Array of indices of matched subhalos in snap_search (snap_ref).
    #
    #     Notes
    #     -----
    #     A subhalo in snap_ref is only matched with one subhalo in
    #     snap_srch, but the same is not true of subhalos in snap_srch.
    #     I.e. the identification from ref to exp can be described as  a
    #     function that is not necessarily injective. This feature is
    #     logically inherited from the is_a_match method.
    #     """
    #
    #     mass_ref = snap_ref.get_subhalos('MassType')[:, 1]
    #     ids_ref = snap_ref.get_subhalos_IDs(part_type=[1])
    #     mass_srch = snap_srch.get_subhalos('MassType')[:, 1]
    #     ids_srch = snap_srch.get_subhalos_IDs(part_type=[1])
    #
    #     matches_ref = self.no_match * np.ones(mass_ref.size, dtype=int)
    #     matches_srch = self.no_match * np.ones(mass_srch.size, dtype=int)
    #     mask_unmatched_srch = np.full(mass_srch.size, True)
    #
    #     sorting_ref = np.argsort(-mass_ref)
    #     sorting_srch = np.argsort(-mass_srch)
    #
    #     # This array tells you, in a mass-sorted array, at which index place
    #     # each subhalo would appear:
    #     mass_ranking_srch = np.argsort(sorting_srch)
    #
    #     # Iterate over subhalos in ref, in descending order by mass:
    #     for sub_idx_ref in sorting_ref:
    #         # Select match candidates from snap_search:
    #         mask_mass_range = self.select_by_mass(
    #             mass_ref[sub_idx_ref], mass_srch
    #         )
    #
    #         # Look for matches among unmatched, in the given mass range:
    #         rank_of_searched = mass_ranking_srch[
    #             np.logical_and(mask_unmatched_srch, mask_mass_range)
    #         ]
    #
    #         # Iterate over the selected subhalos in srch, in descending order
    #         # by mass:
    #         searched_subs = sorting_srch[rank_of_searched]
    #         for sub_idx_srch in searched_subs:
    #             # Match:
    #             found_match = self.is_a_match(ids_ref[sub_idx_ref],
    #                                           mass_ref[sub_idx_ref],
    #                                           ids_srch[sub_idx_srch],
    #                                           mass_srch[sub_idx_srch])
    #
    #             if found_match:
    #                 matches_ref[sub_idx_ref] = sub_idx_srch
    #                 matches_srch[sub_idx_srch] = sub_idx_ref
    #                 mask_unmatched_srch[sub_idx_srch] = False
    #                 break
    #
    #     return matches_ref, matches_srch

    def match_non_injective(self, snap_ref, snap_srch):

        mass_ref = snap_ref.get_subhalos('MassType')[:, 1]
        ids_ref = snap_ref.get_subhalos_IDs(part_type=[1])
        mass_srch = snap_srch.get_subhalos('MassType')[:, 1]
        ids_srch = snap_srch.get_subhalos_IDs(part_type=[1])

        # IDEA: Sort subhalos in snap_search by mass to speed up matching?
        # mass_argsort_srch = np.argsort(search['Mass'])
        # mass_sort_srch = search['Mass'][mass_argsort_srch]

        matches_ref = self.no_match * np.ones(mass_ref.size, dtype=int)

        # Iterate over subhalos in ref:
        for sub_idx_ref in range(mass_ref.size):
            # Select match candidates from snap_search:
            mask_mass_range = self.select_by_mass(
                mass_ref[sub_idx_ref], mass_srch
            )

            # Look for matches in the given mass range:
            searched_subs = np.arange(mass_srch.size)[mask_mass_range]
            for sub_idx_srch in searched_subs:
                # Match:
                found_match = self.is_a_match(ids_ref[sub_idx_ref],
                                              mass_ref[sub_idx_ref],
                                              ids_srch[sub_idx_srch],
                                              mass_srch[sub_idx_srch])

                if found_match:
                    matches_ref[sub_idx_ref] = sub_idx_srch
                    break

        # Invert matches_ref into an array of matches for subhalos in search:
        matches_srch = self.invert_matches(matches_ref, mass_ref, mass_srch)

        return matches_ref, matches_srch

    def is_a_match(self, ids_ref, mass_ref, ids_srch, mass_srch):
        """ Check if two halos with given IDs and masses correspond to the same
        halo.

        Parameters
        ----------
        ids_ref : ndarray of int
            Particle IDs of the reference halo in ascending order by binding
            energy (most bound first).
        mass_ref : float
            Mass of (particles of) first halo.
        ids_srch : ndarray of int
            Particle IDs of the explored halo in ascending order by binding
            energy (most bound first).
        mass_srch : float
            Mass of (particles of) second halo.

        Returns
        -------
        found_match : bool
            True iff halos match.

        Notes
        -----
        Subhalos S_ref and S_srch are linked, if
            1. their total DM masses are within the fraction ´f_mass_link´
            from each other, and
            2. more than half of the ´n_link_ref´ most bound DM particles of
            S_ref are among the max(n_srch * ´f_link_srch´, ´n_link_ref´) most
            bound DM particles of S_srch (where n_srch is the DM particle
            number of S_srch).

        The reference subhalo can only be matched with one subhalo in the
        explored dataset. Explicitly, this is because of the following: if more
        than half of the n_link most bound particles in the reference subhalo
        are found in a given halo S, then there can be no other subhalo
        satisfying the same condition (no duplicates of any ID in dataset).
        """

        found_match = False

        # Ignore pure gas halos:
        if mass_ref > 0 and mass_srch > 0:

            # Check masses:
            if (mass_ref / mass_srch < self.f_mass_link) and \
                    (mass_ref / mass_srch > 1 / self.f_mass_link):

                # Get most bound particles:
                most_bound_ref = ids_ref[:self.n_link_ref]
                most_bound_srch = ids_srch[:max(
                    int(ids_srch.size * self.f_link_srch),
                    self.n_link_ref
                )]

                # Check intersection:
                shared_parts = np.intersect1d(most_bound_ref,
                                              most_bound_srch,
                                              assume_unique=True)
                if len(shared_parts) > self.n_link_ref / 2:
                    found_match = True

        return found_match

    def select_by_mass(self, mass_ref, mass_array):
        """ Select elements from array that are within a given factor from
        the reference mass. """

        mask = np.logical_and(
            mass_array > mass_ref / self.f_mass_link, # massive enough
            mass_array < mass_ref * self.f_mass_link # not too massive
        )

        return mask

    def invert_matches(self, matches_ref, mass_ref, mass_srch):

        # Initialize match array for subhalos in search:
        matches_srch = self.no_match * np.ones(
            (mass_srch.size, self.n_matches), dtype=int)

        # Iterate through matches:
        for sub_idx_ref, sub_idx_srch in enumerate(matches_ref):
            if sub_idx_srch == self.no_match:
                continue

            # Iterate through matches of subhalo in ref (at index place
            # sub_idx_ref):
            for i, match_srch in enumerate(matches_srch[sub_idx_srch]):

                # No match yet listed:
                if match_srch == self.no_match:
                    matches_srch[sub_idx_srch, i] = sub_idx_ref
                    break

                # A match listed, but less massive:
                elif mass_ref[sub_idx_ref] > mass_ref[match_srch]:
                    print('here')
                    # Insert the new match in place:
                    matches_srch[sub_idx_srch] = np.insert(
                        matches_srch[sub_idx_srch], i, sub_idx_ref
                    )[:self.n_matches]
                    break

        return matches_srch
