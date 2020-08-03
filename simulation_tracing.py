import h5py
import numpy as np
import os.path

import halo_matching


class SimulationTracer:

    def __init__(self, snap_z0, no_match=2 ** 32):
        """

        Parameters
        ----------
        snap_z0 : Snapshot
            Simulation snapshot at redshift z=0.
        """
        self.sim_id = snap_z0.sim_id
        self.n_halos = snap_z0.get_halo_number()
        self.earliest_snap = snap_z0.snap_id
        self.no_match = no_match

        # Name of file for saving the tracer:
        self.tracer_file = ".tracer_{}.hdf5".format(snap_z0.sim_id)

        # If the tracer file does not yet exists, create it:
        if not os.path.isfile(self.tracer_file):
            print('creating file {}'.format(self.tracer_file))
            # Initialize tracer:
            tracer_arr = np.ones((self.n_halos, snap_z0.snap_id + 1),
                                 dtype=int) * no_match

            # Identify halos in snap_z0 with themselves:
            tracer_arr[:, snap_z0.snap_id] = np.arange(self.n_halos)

            # Add tracer array to file:
            with h5py.File(self.tracer_file, "w") as f:
                tracer_dset = f.create_dataset("tracer", data=tracer_arr)
                tracer_dset.attrs['earliest_snap'] = snap_z0.snap_id
                tracer_dset.attrs['no_match'] = no_match

                # For convenience, write snapshot ids as well:
                f.create_dataset("snapshot_ids", data=np.arange(
                    snap_z0.snap_id + 1))
        else:
            with h5py.File(self.tracer_file, 'r') as f:
                self.earliest_snap = f['tracer'].attrs.get(
                    'earliest_snap')

    def trace_all(self, stop):
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

        # Get the earliest traced snapshot and continue from there:
        with h5py.File(self.tracer_file, "r") as f:
            tracer = f['tracer']
            no_match = tracer.attrs.get('no_match')
            snap = Snapshot(self.sim_id, tracer.attrs.get('earliest_snap'))
            tracer = tracer[...]
            snap_ids = f['snapshot_ids'][...]

        while snap.snap_id > stop:
            snap_next = Snapshot(self.sim_id, snap.snap_id - 1)
            matches = halo_matching.match_snapshots(snap, snap_next,
                                                    no_match)

            # Get those matches that connect to redshift zero:
            for z0_idx, prev_idx in enumerate(tracer[:, snap.snap_id]):
                if prev_idx != no_match:
                    tracer[z0_idx, snap_next.snap_id] = matches[prev_idx]

            # Save new matches:
            with h5py.File(self.tracer_file, "r+") as f:
                f['tracer'][:, snap_next.snap_id] = tracer[:,
                                                         snap_next.snap_id]
                f['tracer'].attrs.modify('earliest_snap',
                                         snap_next.snap_id)
            self.earliest_snap = snap.snap_id

            snap = snap_next

        return tracer[:, stop:], snap_ids[stop:]

    def get_tracer(self):

        with h5py.File(self.tracer_file, 'r') as f:
            out = f['tracer'][:, self.earliest_snap:]

        return out

    def get_snapshot_ids(self):

        with h5py.File(self.tracer_file, 'r') as f:
            out = f['snapshot_ids'][self.earliest_snap:]

        return out

    def get_redshifts(self):

        z = np.array([snapshot_obj.Snapshot(self.sim_id, snap_id)
                      for snap_id in self.get_snapshot_ids()])

        return z