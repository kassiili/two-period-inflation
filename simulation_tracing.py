import numpy as np
import os.path

from snapshot_obj import Snapshot
import halo_matching


class SimulationTracer:

    def __init__(self, snap_z0, stop):
        """

        Parameters
        ----------
        snap_z0 : Snapshot
            Simulation snapshot at redshift z=0.
        stop : int
            The id of the snapshot beyond which the tracing is not
            continued.
        """
        self.snap_z0 = snap_z0
        self.stop = stop
        n_halos = snap_z0.get_halo_number()

        # Name of file for saving the tracer:
        self.tracer_file = ".tracer_{}_from{}to{}.npy".format(
            snap_z0.sim_id, snap_z0.snap_id, self.stop)

        # Set value indicating no match:
        self.no_match = 2 ** 32

        # Initialize tracer:
        self.tracer = np.ones((n_halos, snap_z0.snap_id + 1),
                              dtype=int) * self.no_match
        # Identify halos in snap_ref with themselves:
        self.tracer[:, snap_z0.snap_id] = np.arange(n_halos)

    def trace_all(self):
        """ Traces all subhalos of given galaxies as far back in time as
        possible, starting from the given snapshot.

        Returns
        -------
        tracer : ndarray of lists
            The lists trace the indices of the subhalos through snapshots.
            Each list element is a tuple, where the first entry is the
            snap_id and the second entry is the idx of the subhalo in that
            snapshot.
        """

        # Check if tracer file already exists:
        if os.path.isfile(self.tracer_file):
            self.tracer = np.load(self.tracer_file)

        else:
            snap = self.snap_z0
            while snap.snap_id > self.stop:
                snap_next = Snapshot(snap.sim_id, snap.snap_id - 1)
                matches = halo_matching.match_snapshots(snap, snap_next,
                                                    no_match=self.no_match)
                for halo_tracer in self.tracer:
                    last_idx = halo_tracer[snap.snap_id]
                    if last_idx != self.no_match:
                        halo_tracer[snap_next.snap_id] = matches[last_idx]

                snap = snap_next

            # Save tracer:
            np.save(self.tracer_file, self.tracer)

        return self.tracer
