import numpy as np

import dataset_compute
import simulation_tracing
from snapshot_obj import Snapshot


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
            self.gn = self.get_halo_data('GroupNumber')
            self.sgn = self.get_halo_data('SubGroupNumber')

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
        halo_cop = dataset_compute.periodic_wrap(Snapshot(
            self.sim_id, snap_id), galactic_centre, halo_cop)

        return halo_cop - galactic_centre


class SubhaloTracer:

    def __init__(self, simulation, snap_id_ref, gn_ref, sgn_ref):
        """

        Parameters
        ----------
        simulation : Simulation object
        """
        self.simulation = simulation

        # Initialize tracer:
        snap_ref = self.simulation.get_snapshot(snap_id_ref)
        self.tracer = [[SubhaloInstance(snap_ref, gn=gn_ref,
                                        sgn=sgn_ref)], [snap_id_ref]]

    def trace(self, merger_tree):
        subh_idx_ref = self.tracer[0][0].get_index()
        snap_id_ref = self.tracer[1][0]
        #print(snap_id_ref, subh_idx_ref)

        # Find indices of descendants in their respective snapshots:
        desc_idx = simulation_tracing.find_subhalo_descendants(
            merger_tree, snap_id_ref, subh_idx_ref)[1:]
        snap_ids = list(range(snap_id_ref + 1,
                              snap_id_ref + 1 + len(desc_idx)))

        # Generate SubhaloInstances of descendants:
        descendants = []
        for snap_id, subh_index in zip(snap_ids, desc_idx):
            #print(snap_id, subh_index)
            descendants.append(SubhaloInstance(
                self.simulation.get_snapshot(snap_id), idx=subh_index))

        # Add descendants to tracer:
        self.tracer[0] = self.tracer[0] + descendants
        self.tracer[1] = self.tracer[1] + snap_ids

        # Find indices of progenitors in their respective snapshots:
        prog_idx = simulation_tracing.find_subhalo_progenitors(
            merger_tree, snap_id_ref, subh_idx_ref)[:0:-1]
        snap_ids = list(range(snap_id_ref - len(prog_idx),
                              snap_id_ref))

        # Generate SubhaloInstances of progenitors:
        progenitors = []
        for snap_id, subh_index in zip(snap_ids, prog_idx):
            #print(snap_id, subh_index)
            progenitors.append(SubhaloInstance(
                self.simulation.get_snapshot(snap_id), idx=subh_index))

        # Add progenitors to tracer:
        self.tracer[0] = progenitors + self.tracer[0]
        self.tracer[1] = snap_ids + self.tracer[1]

    def get_halo_data(self, data_name, snap_start=None, snap_stop=None):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        # If neither limit is given, return all snapshots:
        if snap_start is None and snap_stop is None:
            idx_start = 0
            idx_stop = len(self.tracer[1])
            data = np.array([shi.get_halo_data(data_name) for shi in
                             self.tracer[0][idx_start:idx_stop]])

        # If only one is given, return only that snapshot:
        elif snap_start is None:
            idx = self.tracer[1].index(snap_stop)
            data = self.tracer[0][idx].get_halo_data(data_name)
        elif snap_stop is None:
            idx = self.tracer[1].index(snap_start)
            data = self.tracer[0][idx].get_halo_data(data_name)

        # If both are given, return snapshots between the limits:
        else:
            idx_start = self.tracer[1].index(snap_start)
            idx_stop = self.tracer[1].index(snap_stop - 1) + 1
            data = np.array([shi.get_halo_data(data_name) for shi in
                             self.tracer[0][idx_start:idx_stop]])

        return data

    def get_ids(self, part_type):
        """ Retrieves the bound particle IDs in the given snapshot.
        """

        ids = np.array([shi.get_ids(part_type) for shi in self.tracer[0]])

        return ids

    def distance_to_central(self, central_tracer, snap_start=None,
                            snap_stop=None):
        """ Compute distance to the central galaxy at the given snapshot.
        """

        if snap_start is None or snap_start < min(self.tracer[1]):
            snap_start = min(self.tracer[1])
        if snap_stop is None or snap_stop > max(self.tracer[1]):
            snap_stop = max(self.tracer[1]) + 1

        print(snap_start, snap_stop)
        halo_cop = self.get_halo_data("CentreOfPotential", snap_start,
                                      snap_stop)
        galactic_centre = central_tracer.get_halo_data(
            "CentreOfPotential", snap_start, snap_stop)

        # Wrap around centre:
        halo_cop = np.array([dataset_compute.periodic_wrap(
            self.simulation.get_snapshot(sid), ctr, cop)
            for sid, ctr, cop in
            zip(self.tracer[1], galactic_centre, halo_cop)])

        return halo_cop - galactic_centre
