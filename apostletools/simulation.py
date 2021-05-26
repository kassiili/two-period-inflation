import numpy as np
from collections.abc import Iterable

from snapshot_obj import Snapshot
import datafile_oper


class Simulation:

    def __init__(self, sim_id, sim_path=None):
        self.sim_id = sim_id
        if sim_path is None:
            self.sim_path = ""
        else:
            self.sim_path = sim_path
        self.snapshots = self.create_snapshots()
        self.m31 = (-1, -1)
        self.mw = (-1, -1)

    def create_snapshots(self):
        """ Create a dictionary of snapshot objects with snapshot
        identifiers as keys. """
        snap_ids = self.get_snap_ids()
        snapshots = {snap_id: Snapshot(self.sim_id, snap_id,
                                       sim_path=self.sim_path)
                     for snap_id in snap_ids}
        return snapshots

    def set_centrals(self, m31, mw):
        self.m31 = m31
        self.mw = mw

    def get_subhalos_in_snapshots(self, snap_ids, dataset, group='Subhalo'):
        data = [self.snapshots[snap_id].get_subhalos(dataset, group)
                for snap_id in snap_ids]

        return data

    def compute_in_snapshots(self, snap_ids, func, *args):
        data = [func(self.snapshots[snap_id], args)[:,0] for snap_id in
                snap_ids]

        return data

    def get_snap_ids(self):
        return datafile_oper.get_snap_ids(self.sim_id)

    def get_snapshots(self, snap_start, snap_stop):
        snaps = np.array([self.snapshots[snap_id] for snap_id in
                          range(snap_start, snap_stop)])

        return snaps

    def get_snapshot(self, snap_id):
        try:
            snap = self.snapshots[snap_id]
            return snap
        except KeyError:
            # print("No entry for snapshot {}".format(snap_id) + "in
            # the specified location for the simulation data: " + "{
            # }".format(self.sim_path))
            return None

    def get_snapshots_in_array(self):
        return self.snapshots.values()

    def get_snap_num(self):
        return max(self.snapshots) + 1

    def get_redshifts(self, snap_start=None, snap_stop=None):

        if snap_start is None:
            snap_start = min(self.get_snap_ids())
        if snap_stop is None:
            snap_stop = max(self.get_snap_ids()) + 1

        z = np.array([self.snapshots[snap_id].get_attribute("Redshift",
                                                            "Header")
                      for snap_id in range(snap_start, snap_stop)])

        return z

    def get_hubble(self, snap_start=None, snap_stop=None):

        if snap_start is None:
            snap_start = min(self.get_snap_ids())
        if snap_stop is None:
            snap_stop = max(self.get_snap_ids()) + 1

        hubble = np.array([self.snapshots[snap_id].get_attribute("H(z)",
                                                                 "Header")
                          for snap_id in range(snap_start, snap_stop)])

        return hubble