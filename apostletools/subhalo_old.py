import numpy as np

import dataset_comp
import simtrace_old
from snapshot import Snapshot


class SubhaloInstance:

    def __init__(self, snap, idx=None, gn=None, sgn=None):
        """

        Parameters
        ----------
        snap : Snapshot
        """
        self.snap = snap

        # Assume that either (gn,sgn) or idx is given:
        self.gn = gn
        self.sgn = sgn
        self.idx = idx
        if self.idx is None:
            self.idx = self.get_index()
        if self.gn is None or self.sgn is None:
            self.gn = int(self.get_halo_data('GroupNumber'))
            self.sgn = int(self.get_halo_data('SubGroupNumber'))
        #print("Created subhalo instance: {}, {} at {}".format(self.gn,
        #                                                      self.sgn,
        #                                                      snap.snap_id))

    def get_index(self):

        gns = self.snap.get_subhalos("GroupNumber")
        sgns = self.snap.get_subhalos("SubGroupNumber")
        idx = np.nonzero(np.logical_and(gns == self.gn,
                                        sgns == self.sgn))[0][0]

        return idx

    def get_halo_data(self, data_name):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        data = self.snap.get_subhalos(data_name)[self.idx]
        return data

    def get_fof_data(self, data_name):
        # The FOF datasets are ordered by ascending group number (with index
        # 0 having gn 1):
        data = self.snap.get_subhalos(data_name, 'FOF')[self.gn - 1]
        return data

    def find_group_numbers(self):

        if not self.gn and not self.sgn:
            self.gn = self.get_halo_data('GroupNumber')
            self.sgn = self.get_halo_data('SubGroupNumber')

        return self.gn, self.sgn

    def get_ids(self, part_type):
        """ Retrieves the bound particle IDs in the given snapshot.
        """

        ids = self.snap.get_subhalos_IDs(part_type=part_type)[self.idx]
        return ids

    def distance_to_central(self, snap_id, central):
        """ Compute distance to the central galaxy at the given snapshot.
        """

        galactic_centre = central.get_halo_data("CentreOfPotential",
                                                snap_id)
        halo_cop = self.get_halo_data("CentreOfPotential", snap_id)
        halo_cop = dataset_comp.periodic_wrap(Snapshot(
            self.sim_id, snap_id), galactic_centre, halo_cop)

        return halo_cop - galactic_centre


class SubhaloTracer:

    def __init__(self, simulation, snap_id_ref, gn_ref=None,
                 sgn_ref=None, idx_ref=None):
        """

        Parameters
        ----------
        simulation : Simulation object
        """
        self.simulation = simulation

        # Initialize tracer:
        snap_ref = self.simulation.get_snapshot(snap_id_ref)

        self.tracer = np.array([None for i in range(simulation.get_snap_num())])
        self.tracer[snap_id_ref] = SubhaloInstance(snap_ref, idx=idx_ref,
                                                   gn=gn_ref, sgn=sgn_ref)

    def trace(self, merger_tree):
        snap_id_ref = [instance is not None for instance in
                       self.tracer].index(True)
        subh_idx_ref = self.tracer[snap_id_ref].get_index()

        # Find indices of descendants in their respective snapshots:
        desc_idx = simtrace_old.find_subhalo_descendants(
            merger_tree, snap_id_ref, subh_idx_ref)[1:]
        snap_ids = list(range(snap_id_ref + 1,
                              snap_id_ref + 1 + len(desc_idx)))
        #print("Got descendant indices")

        # Generate SubhaloInstances of descendants:
        descendants = []
        for snap_id, subh_index in zip(snap_ids, desc_idx):
            #print(snap_id, subh_index)
            descendants.append(SubhaloInstance(
                self.simulation.get_snapshot(snap_id), idx=subh_index))

        # Add descendants to tracer:
        down = snap_id_ref + 1
        up = down + len(descendants)
        self.tracer[down:up] = descendants

        # Find indices of progenitors in their respective snapshots:
        prog_idx = simtrace_old.find_subhalo_progenitors(
            merger_tree, snap_id_ref, subh_idx_ref)[:0:-1]
        snap_ids = list(range(snap_id_ref - len(prog_idx),
                              snap_id_ref))
        #print("Got progenitor indices")

        # Generate SubhaloInstances of progenitors:
        progenitors = []
        for snap_id, subh_index in zip(snap_ids, prog_idx):
            #print(snap_id, subh_index)
            progenitors.append(SubhaloInstance(
                self.simulation.get_snapshot(snap_id), idx=subh_index))

        # Add progenitors to tracer:
        self.tracer[snap_id_ref - len(progenitors):snap_id_ref] = \
            progenitors

    def get_identifier(self, snap):
        gn = self.tracer[snap].gn
        sgn = self.tracer[snap].sgn

        return gn, sgn

    def get_halo_data(self, data_name, snap_start=None, snap_stop=None):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        # If neither limit is given, return all snapshots:
        if snap_start is None and snap_stop is None:
            data = np.array([shi.get_halo_data(data_name) for shi in
                             self.tracer if shi is not None])

        # If only one is given, return only that snapshot:
        elif snap_start is None:
            data = self.tracer[snap_stop].get_halo_data(data_name)
        elif snap_stop is None:
            data = self.tracer[snap_start].get_halo_data(data_name)

        # If both are given, return snapshots between the limits:
        else:
            data = np.array([shi.get_halo_data(data_name)
                             for shi in self.tracer[snap_start:snap_stop]
                             if shi is not None])

        return data

    def get_fof_data(self, data_name, snap_start=None, snap_stop=None):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        # If neither limit is given, return all snapshots:
        if snap_start is None and snap_stop is None:
            data = np.array([shi.get_fof_data(data_name) for shi in
                             self.tracer if shi is not None])

        # If only one is given, return only that snapshot:
        elif snap_start is None:
            data = self.tracer[snap_stop].get_fof_data(data_name)
        elif snap_stop is None:
            data = self.tracer[snap_start].get_fof_data(data_name)

        # If both are given, return snapshots between the limits:
        else:
            data = np.array([shi.get_fof_data(data_name)
                             for shi in self.tracer[snap_start:snap_stop]
                             if shi is not None])

        return data

    def get_ids(self, part_type):
        """ Retrieves the bound particle IDs in the given snapshot.
        """

        ids = np.array([shi.get_ids(part_type) for shi in self.tracer[0]])

        return ids

    def get_traced_snapshots(self):
        traced_snaps = np.nonzero(
            [instance is not None for instance in self.tracer])[0]

        return traced_snaps

    def distance_to_central(self, central_tracer, snap_start=None,
                            snap_stop=None, centre_name=None):
        """ Compute distance to the central galaxy at the given snapshot.
        """

        min_snap = min(self.get_traced_snapshots())
        max_snap = max(self.get_traced_snapshots())
        if snap_start is None or snap_start < min_snap:
            snap_start = min_snap
        if snap_stop is None or snap_stop > max_snap:
            snap_stop = max_snap + 1
        if centre_name is None:
            centre_name = "CentreOfPotential"

        #print(snap_start, snap_stop)
        halo_cop = self.get_halo_data(centre_name, snap_start,
                                      snap_stop)
        galactic_centre = central_tracer.get_halo_data(
            centre_name, snap_start, snap_stop)

        # Wrap around centre:
        halo_cop = np.array([dataset_comp.periodic_wrap(
            self.simulation.get_snapshot(sid), ctr, cop)
            for sid, ctr, cop in
            zip(np.arange(snap_start, snap_stop), galactic_centre, halo_cop)])

        return galactic_centre - halo_cop
