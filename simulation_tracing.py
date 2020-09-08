import h5py
import numpy as np

import data_file_manipulation
import halo_matching


def find_subhalo_descendants(merger_tree, snap_id, index_of_subhalo):
    tracer = merger_tree.get_all_matches()

    # Get descendants from reference snapshot:
    descendant_idxs = []
    idx = index_of_subhalo
    while idx != merger_tree.no_match:
        descendant_idxs.append(idx)
        if 'Descendants' not in tracer[snap_id].keys():
            break
        idx = tracer[snap_id]['Descendants'][idx]
        snap_id += 1

    return descendant_idxs


def find_subhalo_progenitors(merger_tree, snap_id, index_of_subhalo):
    tracer = merger_tree.get_all_matches()

    # Get descendants from reference snapshot:
    progenitor_idxs = []
    idx = index_of_subhalo
    while idx != merger_tree.no_match:
        progenitor_idxs.append(idx)
        if 'Progenitors' not in tracer[snap_id].keys():
            break
        idx = tracer[snap_id]['Progenitors'][idx, 0]
        snap_id -= 1

    return progenitor_idxs


class SnapshotTracer:
    """ Traces the subhalos in a single snapshot forward and backward
    in time.

    Notes
    -----
    When tracing in the direction, where the merger tree allows
    branching, the family lines of the most massive relatives are
    followed.
    """

    def __init__(self, snap_id, merger_tree):
        self.merger_tree = merger_tree
        self.no_match = merger_tree.no_match
        self.snap_id = snap_id
        n_halos = merger_tree.simulation.get_snapshot(
            snap_id).get_subhalo_number()
        n_snaps = merger_tree.simulation.get_snap_num()
        self.tracer_array = merger_tree.no_match * \
                            np.ones((n_halos, n_snaps), dtype=int)
        self.tracer_array[:, snap_id] = np.arange(n_halos)
        self.traced_snaps = [snap_id, snap_id + 1]

    def trace(self, start=None, stop=None):

        n_snaps = np.size(self.tracer_array, axis=1)

        if start is None and stop is None:
            start = 0
            stop = n_snaps
        elif start is None or stop is None:
            if start is None:
                lim = stop
            else:
                lim = start
            start = min(self.snap_id, lim)
            stop = max(self.snap_id + 1, lim)

        self.trace_forward(stop)
        self.trace_backward(start)

        out = self.tracer_array[:, start:stop]

        return out

    def trace_forward(self, stop):

        n_halos = np.size(self.tracer_array, axis=0)

        # Set starting point to last untracked snapshot:
        start = self.traced_snaps[1] - 1
        heritage = self.merger_tree.get_all_matches()

        for sid in range(start + 1, stop):
            # Indices of the traced subhalos in the previous snapshot:
            idx_in_prev = self.tracer_array[:, sid - 1]

            # Indices of subhalos in the previous snapshot, in the
            # current snapshot:
            idx_from_prev = heritage[sid - 1]
            idx_from_prev = idx_from_prev['Descendants']

            # To avoid tracer out of bounds errors,
            # fill insufficiently short tracer arrays with no_match:
            if idx_from_prev.size < n_halos:
                fill = self.no_match * \
                       np.ones(n_halos - idx_from_prev.size)
                idx_from_prev = np.append(idx_from_prev, fill)

            # To avoid out of bounds errors, replace values that are
            # equal to no_match:
            help_idx = np.where(idx_in_prev != self.no_match,
                                idx_in_prev, np.arange(n_halos))

            idx_from_start = idx_from_prev[help_idx]

            # Save new indices of subhalos that were traced up to
            # the previous snapshot:
            self.tracer_array[:, sid] = np.where(
                idx_in_prev != self.no_match, idx_from_start,
                self.no_match)

        self.traced_snaps[1] = max(self.traced_snaps[1], stop)

    def trace_backward(self, start):

        n_halos = np.size(self.tracer_array, axis=0)

        # Set stopping point to first tracked snapshot:
        stop = self.traced_snaps[0]
        heritage = self.merger_tree.get_all_matches()

        for sid in range(stop - 1, start - 1, -1):
            # Indices of the traced subhalos in the following snapshot:
            idx_in_foll = self.tracer_array[:, sid + 1]

            # Indices of subhalos in the following snapshot, in the
            # current snapshot:
            idx_from_foll = heritage[sid + 1]
            idx_from_foll = idx_from_foll['Progenitors'][:, 0]

            # To avoid tracer out of bounds errors,
            # fill insufficiently short tracer arrays with no_match:
            if idx_from_foll.size < n_halos:
                fill = self.no_match * \
                       np.ones(n_halos - idx_from_foll.size)
                idx_from_foll = np.append(idx_from_foll, fill)

            # To avoid out of bounds errors, replace values that are
            # equal to no_match:
            help_idx = np.where(idx_in_foll != self.no_match,
                                idx_in_foll, np.arange(n_halos))

            idx_from_start = idx_from_foll[help_idx]

            # Save new indices of subhalos that were traced up to
            # the previous snapshot:
            self.tracer_array[:, sid] = np.where(
                idx_in_foll != self.no_match, idx_from_start,
                self.no_match)

        self.traced_snaps[0] = min(self.traced_snaps[0], start)

    def get_indices_at_snapshot(self, snap_id):
        return self.tracer_array[:, snap_id]

    def find_creation(self):
        creation_snaps = -1 * np.ones(np.size(self.tracer_array, axis=0))
        self.trace_backward(0)

        for snap_id in range(self.snap_id + 1):
            # Get subhalos that existed at snap and did not exist at an
            # earlier snap:
            mask_created_at_snap = np.logical_and(
                self.tracer_array[:, snap_id] != self.no_match,
                creation_snaps == -1
            )
            creation_snaps = np.where(mask_created_at_snap, snap_id,
                                      creation_snaps)

        return creation_snaps


class SatelliteTracer:

    def __init__(self, merger_tree, satellite_selector):
        self.merger_tree = merger_tree
        self.satellite_selector = satellite_selector
        self.tracer_arrays = []

    def trace(self):
        heritage = self.merger_tree.get_all_matches()
        snap_ids = list(heritage.keys()).sort()
        for snap_id in snap_ids:
            pass


class MergerTree:

    def __init__(self, simulation, matcher=None,
                 branching='BackwardBranching',
                 min_snaps_traced=1):
        """

        Parameters
        ----------
        branching : str
            The merger tree only allows branching in one direction (in
            time), which is indicated by this value.
        """
        self.simulation = simulation
        if matcher is None:
            self.matcher = halo_matching.SnapshotMatcher()
        else:
            self.matcher = matcher
        self.no_match = self.matcher.no_match
        self.branching = branching
        self.min_snaps_traced = min_snaps_traced
        self.storage_file = '.tracer_{}_{}.hdf5'.format(
            self.branching, self.simulation.sim_id)

    def build_tree(self, snap_id_1, snap_id_2):
        """ Find descendants and progenitors of all subhalos between
        given snapshots.
        """

        if self.branching == 'ForwardBranching':
            self.build_tree_with_forward_branch(snap_id_1, snap_id_2)
        else:
            self.build_tree_with_back_branch(snap_id_1, snap_id_2)

    def build_tree_with_back_branch(self, snap_id_1, snap_id_2):
        """" Find subhalo heritage iterating forward in time.

        Notes
        -----
        In this case, when two subsequent snapshots are compared,
        the subhalos in the earlier snapshot will only be matched with
        a single subhalo in the next snapshot, so that branching only
        happens 'backward' in time. """

        snap_start = min(snap_id_1, snap_id_2)
        snap_stop = max(snap_id_1, snap_id_2)

        # Get the first snapshot:
        snap = self.simulation.get_snapshot(snap_start)

        while snap.snap_id < snap_stop - 1:
            # Get next snapshot for matching:
            snap_next = self.get_next_snap(snap.snap_id)
            if snap_next is None:
                break

            # If matches are already saved, read them - otherwise, do the
            # matching:
            h5_group = 'Extended/Heritage/BackwardBranching'
            desc_exists = data_file_manipulation.group_dataset_exists(
                snap, 'Descendants', h5_group)
            prog_next_exists = \
                data_file_manipulation.group_dataset_exists(
                    snap_next, 'Progenitors', h5_group)

            # Find descendants and progenitors:
            if not desc_exists or not prog_next_exists:
                descendants, progenitors_next = \
                    self.matcher.match_snapshots(snap, snap_next)
            else:
                descendants = snap.get_subhalos('Descendants', h5_group)
                progenitors_next = snap_next.get_subhalos('Progenitors',
                                                          h5_group)
            # Save matches to the subhalo catalogues:
            if not desc_exists:
                data_file_manipulation.save_dataset(
                    descendants, 'Descendants', h5_group, snap)
            if not prog_next_exists:
                data_file_manipulation.save_dataset(
                    progenitors_next, 'Progenitors', h5_group, snap_next)

            snap = snap_next

        # Remove connections of volatile subhalos:
        if self.min_snaps_traced > 1:
            self.prune_tree()

    # NOT VERIFIED!
    def build_tree_with_forward_branch(self, snap_id_1, snap_id_2):

        snap_start = max(snap_id_1, snap_id_2)
        snap_stop = min(snap_id_1, snap_id_2)

        # Get the first snapshot:
        snap = self.simulation.get_snapshot(snap_start)

        while snap.snap_id != snap_stop - 1:
            # Get next snapshot for matching:
            snap_next = self.get_next_snap(snap.snap_id)
            if snap_next is None:
                break

            # If matches are already saved, read them - otherwise, do the
            # matching:
            h5_group = 'Extended/Heritage/ForwardBranching'
            progenitors = snap.get_subhalos('Progenitors', h5_group)
            descendants_next = snap_next.get_subhalos('Descendants',
                                                      h5_group)

            if descendants_next.size == 0 or progenitors.size == 0:
                descendants_next, progenitors = \
                    self.matcher.match_snapshots(snap, snap_next)
                # Save matches to the subhalo catalogues:
                data_file_manipulation.save_dataset(
                    descendants_next, 'Descendants', h5_group, snap_next)
                data_file_manipulation.save_dataset(
                    progenitors, 'Progenitors', h5_group, snap)

            snap = snap_next

        # Remove connections of volatile subhalos:
        if self.min_snaps_traced > 1:
            self.prune_tree()

    def get_next_snap(self, cur_snap_id):
        # Set snap_id incrementation value:
        if self.branching == 'ForwardBranching':
            incr = -1
        else:
            incr = 1

        snap_next = self.simulation.get_snapshot(cur_snap_id + incr)

        return snap_next

    def store_tracer(self):
        """ Save another copy of the matches into a NumPy file. """

        match_dict = self.get_all_matches()

        # Save all non-empty entries to the storage file:
        with h5py.File(self.storage_file, 'w') as f:
            for snap_id, matches in match_dict.items():
                h5_group = '/Heritage/{}/{}'.format(self.branching,
                                                    str(snap_id))
                f.create_group(h5_group)
                for key, arr in matches.items():
                    if arr.size != 0:
                        f[h5_group].create_dataset(key, data=arr)

        return match_dict

    def get_all_matches(self):
        """ Return all found matches in a dictionary. """

        # Get snapshot identifiers:
        snap_ids = self.simulation.get_snap_ids()

        # Add all non-empty entries to the output dictionary:
        match_dict = dict()
        for snap_id in snap_ids:
            snap = self.simulation.get_snapshot(snap_id)
            if snap is None:
                continue

            h5_group = 'Extended/Heritage/{}'.format(self.branching)
            descendants = snap.get_subhalos('Descendants', h5_group)
            progenitors = snap.get_subhalos('Progenitors', h5_group)
            if descendants.size != 0 and progenitors.size != 0:
                match_dict[snap_id] = {'Descendants': descendants,
                                       'Progenitors': progenitors}
            elif descendants.size != 0:
                match_dict[snap_id] = {'Descendants': descendants}
            elif progenitors.size != 0:
                match_dict[snap_id] = {'Progenitors': progenitors}

        return match_dict

    def read_from_storage(self):

        match_dict = {}

        with h5py.File(self.storage_file, 'r') as f:
            for snap_id, matches in f['Heritage/{}'.format(
                    self.branching)].items():
                snap_id = int(snap_id)
                match_dict[snap_id] = {}

                snap = self.simulation.get_snapshot(snap_id)
                h5_group = 'Extended/Heritage/{}'.format(self.branching)
                for key, dataset in matches.items():
                    match_dict[snap_id][key] = dataset[...]

                    # Save matches to the subhalo catalogues:
                    data_file_manipulation.save_dataset(
                        dataset[...], key, h5_group, snap)

        return match_dict

    #    def get_matched_snapshots(self):
    #
    #        # Get snapshot identifiers:
    #        snap_ids = self.simulation.get_snap_ids()
    #
    #        matched_snap_ids = set()
    #        for snap_id in snap_ids:
    #            # Get the two potentially matched snapshots:
    #            snap = self.simulation.get_snapshot(snap_id)
    #            snap_next = self.get_next_snap(snap_id)
    #            if snap is None or snap_next is None:
    #                continue
    #
    #            # Get the matches, and if they exist, add to the set of
    #            # matched:
    #            h5_group = "Extended/Matches/{}/{}".format(
    #                self.simulation.sim_id, snap_next.snap_id)
    #            matches = snap.get_subhalos("Matches", h5_group)
    #            if matches.size != 0:
    #                matched_snap_ids.add(snap_id)
    #                matched_snap_ids.add(snap_next.snap_id)
    #
    #        # Get the snapshots, ordered by identifiers:
    #        matched_snaps = np.array(
    #            [self.simulation.get_snapshot(snap_id) for snap_id in
    #             np.sort(list(matched_snap_ids))])
    #
    #        return matched_snaps

    def prune_tree(self):
        # Iterate through the snapshots, keeping track of how many
        # snapshots each subhalo instance extends over. Then remove the
        # connections of subhalos with less than a given amount of
        # connections from the merger tree.
        pass
