import numpy as np

import dataset_compute
from snapshot_obj import Snapshot
import halo_matching


class Subhalo:

    def __init__(self, snap_z0, gn_z0, sgn_z0):
        """

        Parameters
        ----------
        snap_z0 : Snapshot
        """
        self.sim_id = snap_z0.sim_id
        self.tracer = 2**32 * np.ones(snap_z0.snap_id+1, dtype=int)
        self.tracer[snap_z0.snap_id] = snap_z0.index_of_halo(gn_z0, sgn_z0)

    def get_halo_data(self, data_name, snap_id):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        idx = self.tracer[snap_id]
        snap = Snapshot(self.sim_id, snap_id)
        data = snap.get_subhalos(data_name)[idx]

        return data

    def get_particles(self, dataset, snap_id):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        gn, sgn = self.tracer[snap_id]
        snap = Snapshot(self.sim_id, snap_id)
        fnum = snap.file_of_halo(gn, sgn)

        gns = snap.get_particles('GroupNumber', fnums=[fnum])
        sgns = snap.get_particles('SubGroupNumber', fnums=[fnum])
        data = snap.get_particles(dataset, fnums=[fnum]) \
            [np.logical_and((gns == gn), (sgns == sgn))]

        return data

    def get_ids(self, snap_id):
        """ Retrieves the bound particle IDs in the given snapshot.
        """

        gn, sgn = self.tracer[snap_id]
        snap = Snapshot(self.sim_id, snap_id)
        fnum = snap.file_of_halo(gn, sgn)

        # Get index of halo:
        gns = snap.get_subhalos('GroupNumber', fnums=[fnum])
        sgns = snap.get_subhalos('SubGroupNumber', fnums=[fnum])

        ids = snap.get_subhalos_IDs(part_type=1, fnums=[fnum]) \
            [np.logical_and((gns == gn), (sgns == sgn))]

        return ids[0]

    def trace(self, stop=101):
        """ Starting from the earliest identification, trace halo back
        in time as far as possible.

        Parameters
        ----------
        stop : int, optional
            Earliest snapshot to be explored

        Returns
        -------
        tracer : dict of tuple
            Dictionary tracing the gn and sgn values of the halo through
            snapshots. The keys are the snapshot IDs. The corresponding
            redshifts are included as the first element of the tuples (the
            following elements being the gn and the sgn).
        """

        # Get earliest match:
        snap_id = np.min(list(self.tracer.keys()))
        gn, sgn = self.tracer.get(snap_id)
        snap = Snapshot(self.sim_id, snap_id)
        while snap.snap_id > stop:
            snap_next = Snapshot(snap.sim_id, snap.snap_id - 1)
            gn, sgn = halo_matching.find_match(self, snap_id, snap_next)

            # No matching halo found:
            if gn == -1: break

            # Add match to tracer:
            self.tracer[snap_next.snap_id] = (gn, sgn)

            snap = snap_next

        return self.tracer

    def distance_to_central(self, snap_id, central):
        """ Compute distance to the central galaxy at the given snapshot.
        """

        galactic_centre = central.get_halo_data("CentreOfPotential",
                                                snap_id)
        halo_cop = self.get_halo_data("CentreOfPotential", snap_id)
        halo_cop = dataset_compute.periodic_wrap(Snapshot(
            self.sim_id, snap_id), galactic_centre, halo_cop)

        return halo_cop - galactic_centre
