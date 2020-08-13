import h5py
import numpy as np

import snapshot_obj
import data_file_manipulation


class SubhaloTracer:

    def __init__(self, merger_tree, snap_id_ref, gn_ref, sgn_ref):
        self.merger_tree = merger_tree

        # Initialize subhalo index array:
        sim_snap_ids = self.merger_tree.simulation.get_snap_ids()
        self.subhalo_index = np.array(sim_snap_ids.size)
                # Assuming snapshot indexing starts at zero
        self.subhalo_index[:, 0] = sim_snap_ids

        # Set reference index:
        snap_ref = self.merger_tree.simulation.get_snapshot(snap_id_ref)
        idx_ref = snap_ref.index_of_halo(gn_ref, sgn_ref)
        self.subhalo_index[snap_id_ref] = idx_ref

    def trace(self, snap_start, snap_stop):
        tracer = self.merger_tree.trace_all(snap_start, snap_stop)


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

                # To avoid index out of bounds errors,
                # fill insufficiently short index arrays with no_match:
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

    def __init__(self, simulation, matcher, branching="backward",
                 min_snaps_traced=1):
        """

        Parameters
        ----------
        branching : str
            The merger tree only allows branching in one direction (in
            time), which is indicated by this value.
        """
        self.simulation = simulation
        self.matcher = matcher
        self.branching = branching
        self.min_snaps_traced = 1
        self.storage_file = ".tracer_{}_{}.hdf5".format(
            self.branching, self.simulation.sim_id)

    def trace_all(self, snap_id_1, snap_id_2):
        """ Traces all subhalos between snap_id_1 and snap_2. """

        if self.branching == 'backward':
            snap_start = min(snap_id_1, snap_id_2)
            snap_stop = max(snap_id_1, snap_id_2)
        else:
            snap_start = max(snap_id_1, snap_id_2)
            snap_stop = min(snap_id_1, snap_id_2)

        # Get the first snapshot:
        snap = self.simulation.get_snapshot(snap_start)

        # Initialize return value:
        out = dict()

        while snap.snap_id != snap_stop:
            # Get next snapshot for matching:
            snap_next = self.get_next_snap(snap.snap_id)
            if snap_next is None:
                break

            # If matches are already saved, read them - otherwise, do the
            # matching:
            h5_group = 'Extended/Heritage/'
            if self.branching == 'backward':
                h5_group += 'BackwardBranching'
                h5_dataset = 'Descendants'
            else:
                h5_group += 'ForwardBranching'
                h5_dataset = 'Progenitors'
            saved_matches = snap.get_subhalos(h5_dataset, h5_group)

            if saved_matches.size > 0:
                matches = saved_matches
            else:
                matches = self.matcher.match_snapshots(snap, snap_next)
                # Save matches to the subhalo catalogues:
                data_file_manipulation.save_dataset(matches, h5_dataset,
                                                    h5_group, snap)

            # Add matches to the output array:
            out[snap.snap_id] = matches

            snap = snap_next

        # Remove connections of volatile subhalos:
        if self.min_snaps_traced > 1:
            self.prune_tree()

        return out

    def get_next_snap(self, cur_snap_id):
        # Set snap_id incrementation value:
        if self.branching == "backward":
            incr = 1
        else:
            incr = -1

        snap_next = self.simulation.get_snapshot(cur_snap_id + incr)

        return snap_next

    def store_tracer(self):
        """ Save another copy of the matches into a NumPy file. """

        match_dict = self.get_all_matches()

        # Save all non-empty entries to the storage file:
        with h5py.File(self.storage_file, 'w') as f:
            for snap_id, matches in match_dict.items():
                if matches.size != 0:
                    f.create_dataset(str(snap_id), data=matches)

        return match_dict

    def get_all_matches(self):
        """ Return all found matches in a dictionary. """

        # Get snapshot identifiers:
        snap_ids = self.simulation.get_snap_ids()

        # Add all non-empty entries to the output dictionary:
        match_dict = dict()
        for snap_id in snap_ids:
            snap = self.simulation.get_snapshot(snap_id)
            snap_next = self.get_next_snap(snap_id)
            if snap is None or snap_next is None:
                continue
            h5_group = "Extended/Matches/{}/{}".format(
                self.simulation.sim_id, snap_next.snap_id)
            matches = snap.get_subhalos("Matches", h5_group)
            if matches.size != 0:
                match_dict[snap_id] = matches

        return match_dict

    def get_matched_snapshots(self):

        # Get snapshot identifiers:
        snap_ids = self.simulation.get_snap_ids()

        matched_snap_ids = set()
        for snap_id in snap_ids:
            # Get the two potentially matched snapshots:
            snap = self.simulation.get_snapshot(snap_id)
            snap_next = self.get_next_snap(snap_id)
            if snap is None or snap_next is None:
                continue

            # Get the matches, and if they exist, add to the set of
            # matched:
            h5_group = "Extended/Matches/{}/{}".format(
                self.simulation.sim_id, snap_next.snap_id)
            matches = snap.get_subhalos("Matches", h5_group)
            if matches.size != 0:
                matched_snap_ids.add(snap_id)
                matched_snap_ids.add(snap_next.snap_id)

        # Get the snapshots, ordered by identifiers:
        matched_snaps = np.array(
            [self.simulation.get_snapshot(snap_id) for snap_id in
             np.sort(list(matched_snap_ids))])

        return matched_snaps

    def get_redshifts(self):

        z = np.array([snap.get_attribute("Redshift", "Header")
                      for snap in self.get_matched_snapshots()])

        return z

    def prune_tree(self):
        # Iterate through the snapshots, keeping track of how many
        # snapshots each subhalo instance extends over. Then remove the
        # connections of subhalos with less than a given amount of
        # connections from the merger tree.
        pass
