import numpy as np

import dataset_compute
from snapshot_obj import Snapshot
import halo_matching


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

    def get_index(self):

        gns = self.snap.get_subhalos("GroupNumber")
        sgns = self.snap.get_subhalos("SubGroupNumber")
        idx = np.arange(gns.size)[np.logical_and(gns == self.gn,
                                                 sgns == self.sgn)][0]

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

    def __init__(self, gn_z0, sgn_z0, sim_tracer):
        """

        Parameters
        ----------
        sim_tracer : MergerTree
        snap_z0 : Snapshot
        """
        self.sim_id = sim_tracer.sim_id
        snap_ids = sim_tracer.get_snapshot_ids()

        # Get subhalo index at z=0:
        snap_z0 = Snapshot(self.sim_id, snap_ids[-1])
        gns = snap_z0.get_subhalos("GroupNumber")
        sgns = snap_z0.get_subhalos("SubGroupNumber")
        idx_z0 = np.logical_and(gns == gn_z0, sgns == sgn_z0)

        # Create halo tracer:
        halo_idx = sim_tracer.get_tracer_array()[idx_z0][0]
        self.tracer = (np.array([None] * snap_ids.size), snap_ids)
        for i, (hidx, sid) in enumerate(zip(halo_idx, snap_ids)):
            if hidx != sim_tracer.matcher.no_match:
                snap = Snapshot(self.sim_id, sid)
                self.tracer[0][i] = SubhaloInstance(snap, idx=hidx)

        # Restrict to real matches:
        mask = [idx is not None for idx in self.tracer[0]]
        self.tracer = (self.tracer[0][mask], self.tracer[1][mask])

    def get_halo_data(self, data_name):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        data = np.array([shi.get_halo_data(data_name) for shi in
                         self.tracer[0]])

        return data

    def get_ids(self, part_type):
        """ Retrieves the bound particle IDs in the given snapshot.
        """

        ids = np.array([shi.get_ids(part_type) for shi in self.tracer[0]])

        return ids

    def distance_to_central(self, central_tracer):
        """ Compute distance to the central galaxy at the given snapshot.
        """

        halo_cop = self.get_halo_data("CentreOfPotential")
        galactic_centre = central_tracer.get_halo_data("CentreOfPotential")
        galactic_centre = galactic_centre[-np.size(halo_cop, axis=0):]

        # Wrap around centre:
        halo_cop = np.array([dataset_compute.periodic_wrap(
            Snapshot(self.sim_id, sid), ctr, cop) for sid, ctr, cop in
            zip(self.tracer[1], galactic_centre, halo_cop)])

        return halo_cop - galactic_centre
