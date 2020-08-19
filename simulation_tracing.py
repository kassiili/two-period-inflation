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

    def __init__(self, snap_id, merger_tree, direction_in_time="forward"):
        self.merger_tree = merger_tree
        self.snap_id = snap_id
        self.direction = direction_in_time

    def trace(self, stop):
        match_dict = self.merger_tree.trace_all(self.snap_id, stop)
        n_halos = match_dict[self.snap_id].size
        no_match = self.merger_tree.matcher.no_match
        tracer = no_match * np.ones(
            (n_halos, max(self.snap_id, stop) + 1), dtype=int)

        tracer[:, self.snap_id] = np.arange(n_halos)
        if self.direction == "forward":
            for sid in range(self.snap_id + 1, stop + 1):
                # Indices of the traced subhalos in the previous snapshot:
                idx_in_prev = tracer[:, sid - 1]

                # To avoid out of bounds errors, replace values equal
                # to no_match:
                idx_in_prev = np.where(idx_in_prev != no_match,
                                       idx_in_prev, np.arange(n_halos))

                # Indices of subhalos in the previous snapshot, in the
                # current snapshot:
                idx_from_prev = match_dict[sid - 1]

                # To avoid tracer out of bounds errors,
                # fill insufficiently short tracer arrays with no_match:
                if idx_from_prev.size < n_halos:
                    idx_from_prev = np.append(
                        idx_from_prev,
                        no_match * np.ones(n_halos - idx_from_prev.size))

                # Save new indices of subhalos that were traced up to
                # the previous snapshot:
                tracer[:, sid] = np.where(idx_in_prev != no_match,
                                          idx_from_prev[idx_in_prev],
                                          no_match)

        return tracer


class MergerTree:

    def __init__(self, simulation, matcher=None, branching='backward',
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
        if branching == 'forward':
            self.branching = 'ForwardBranching'
        else:
            self.branching = 'BackwardBranching'
        self.min_snaps_traced = min_snaps_traced
        self.storage_file = '.tracer_{}_{}.hdf5'.format(
            self.branching, self.simulation.sim_id)

    def trace_all(self, snap_id_1, snap_id_2):
        """ Find descendants and progenitors of all subhalos between
        given snapshots.
        """

        if self.branching == 'ForwardBranching':
            out = self.trace_all_with_forward_branch(
                snap_id_1, snap_id_2)
        else:
            out = self.trace_all_with_back_branch(
                snap_id_1, snap_id_2)

        return out

    def trace_all_with_back_branch(self, snap_id_1, snap_id_2):
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

        # Initialize return value:
        out = {snap_id: dict() for snap_id in range(snap_start,
                                                    snap_stop)}

        while snap.snap_id != snap_stop - 1:
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

            # Add matches to the output:
            out[snap.snap_id]['Descendants'] = descendants
            out[snap_next.snap_id]['Progenitors'] = progenitors_next

            snap = snap_next

        # Remove connections of volatile subhalos:
        if self.min_snaps_traced > 1:
            self.prune_tree()

        return out

    # NOT VERIFIED!
    def trace_all_with_forward_branch(self, snap_id_1, snap_id_2):

        snap_start = max(snap_id_1, snap_id_2)
        snap_stop = min(snap_id_1, snap_id_2)

        # Get the first snapshot:
        snap = self.simulation.get_snapshot(snap_start)

        # Initialize return value:
        out = {snap_id: dict() for snap_id in range(snap_stop,
                                                    snap_start)}

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

            # Add matches to the output array:
            out[snap_next.snap_id]['Descendants'] = descendants_next
            out[snap.snap_id]['Progenitors'] = progenitors

            snap = snap_next

        # Remove connections of volatile subhalos:
        if self.min_snaps_traced > 1:
            self.prune_tree()

        return out

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
