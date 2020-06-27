import numpy as np

import dataset_compute
from snapshot_obj import Snapshot
import halo_matching


class Subhalo:

    def __init__(self, snap_z0, gn_z0, sgn_z0, sim_tracer):
        """

        Parameters
        ----------
        snap_z0 : Snapshot
        """
        self.sim_id = snap_z0.sim_id

        # Get subhalo tracer:
        gns = snap_z0.get_subhalos("GroupNumber")
        sgns = snap_z0.get_subhalos("SubGroupNumber")
        halo_trace = sim_tracer[0][np.logical_and(gns == gn_z0,
                                                  sgns == sgn_z0)][0]

        self.tracer = (halo_trace, sim_tracer[1])

    def get_halo_data(self, data_name, snap_id):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        idx_trace, snap_ids = self.tracer
        idx = idx_trace[snap_ids == snap_id][0]
        snap = Snapshot(self.sim_id, snap_id)
        data = snap.get_subhalos(data_name)[idx]

        return data

    def get_ids(self, snap_id):
        """ Retrieves the bound particle IDs in the given snapshot.
        """

        idx_trace, snap_ids = self.tracer
        idx = idx_trace[snap_ids == snap_id]
        snap = Snapshot(self.sim_id, snap_id)
        ids = snap.get_subhalos_IDs(part_type=1)[idx]

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
