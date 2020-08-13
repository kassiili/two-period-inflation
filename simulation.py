import numpy as np
from snapshot_obj import Snapshot
import data_file_manipulation


class Simulation:

    def __init__(self, sim_id, sim_path=None):
        self.sim_id = sim_id
        if sim_path is None:
            self.sim_path = ""
        else:
            self.sim_path = sim_path
        self.snapshots = self.create_snapshots()

    def create_snapshots(self):
        """ Create a dictionary of snapshot objects with snapshot
        identifiers as keys. """
        snap_ids = self.get_snap_ids()
        snapshots = [Snapshot(self.sim_id, snap_id,
                              sim_path=self.sim_path)
                     for snap_id in snap_ids]
        snapshots = {snap.snap_id: snap for snap in snapshots}
        return snapshots

    def get_snap_ids(self):
        return data_file_manipulation.get_snap_ids(self.sim_id)

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
