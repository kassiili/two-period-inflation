import numpy as np
from collections.abc import Iterable

from snapshot import Snapshot
from subhalo import Subhalo
import datafile_oper


class Simulation:

    def __init__(self, sim_id, sim_path=None, env_path=""):
        """

        Parameters
        ----------
        sim_id
        sim_path
        env_path : str, optional
            Absolute path for the snapshot data envelope files.
        """
        self.sim_id = sim_id
        if sim_path is None:
            self.sim_path = ""
        else:
            self.sim_path = sim_path
        self.snapshots = self.create_snapshots(env_path)
        self.m31 = (-1, -1)
        self.mw = (-1, -1)

    def create_snapshots(self, env_path):
        """ Create a dictionary of snapshot objects with snapshot
        identifiers as keys. """
        snap_ids = datafile_oper.get_snap_ids(self.sim_id)
        snapshots = {snap_id: Snapshot(
            self.sim_id, snap_id, sim_path=self.sim_path, env_path=env_path
        ) for snap_id in snap_ids}
        return snapshots

    def set_centrals(self, m31, mw):
        self.m31 = m31
        self.mw = mw

    def get_subhalos(self, snap_ids, dset_name, h5_group='Subhalo'):
        data = {snap_id: self.get_snapshot(snap_id).get_subhalos(dset_name,
                                                                 h5_group)
                for snap_id in snap_ids}

        return data

    def compute_in_snapshots(self, snap_ids, func, *args):
        data = [func(self.snapshots[snap_id], args)[:,0] for snap_id in
                snap_ids]

        return data

    def get_snap_ids(self):
        return np.array(list(self.snapshots.keys()))

    def get_snapshots(self, snap_ids=None):
        if snap_ids is None:
            snap_ids = self.get_snap_ids()
        snaps = np.array([self.snapshots[snap_id] for snap_id in snap_ids])

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

    def get_snap_num(self):
        return max(self.snapshots) + 1

    def get_attribute(self, attr_name, entry, snap_ids,
                      data_category='subhalo'):
        """ Reads an attribute of the given entry (either dataset or group)
        of each snapshot.

        Parameters
        ----------
        attr : str
            Attribute name.
        entry : str
            HDF5 group or dataset name with the attribute.
        snap_ids : sequence of int
            Snapshot IDs of snapshot to be read.
        data_category : str, optional
            Specifies, in which datafile the given attribute is read from.

        Returns
        -------
        out : ndarray
            The attribute values.
        """

        try:
            iterator = iter(snap_ids)
        except TypeError:
            # If ´snap_id´ is not iterable (but, for instance, an integer):
            snap_ids = [snap_ids]

        attr = np.array([
            snap.get_attribute(attr_name, entry, data_category=data_category)
            for snap in self.get_snapshots(snap_ids)
        ])

        return attr

    def trace_subhalos(self, snap_start, snap_stop):
        """ Read links of all individual subhalos.

        Parameters
        ----------
        mtree
        snap_start
        snap_stop

        Returns
        -------
        sub_dict : dict of ndarray of Subhalo object
            Contains items for each snapshot, with snapshot IDs as keys, and
            with arrays of subhalos present in a snapshot as values.
        """

        sub_dict = {}

        snap_ids = np.arange(snap_start, snap_stop)
        for sid in snap_ids:
            snapshot = self.get_snapshot(sid)

            if sid == snap_start:
                # Create Subhalo objects of subhalos in the current snapshot and
                # add to ´sub_dict´:
                sub_dict[sid] = np.array([
                    Subhalo(self, i, sid) for i in range(
                        snapshot.get_subhalo_number())
                ])
                continue

            progenitors = snapshot.get_subhalos(
                'Progenitors', h5_group='Extended/Heritage/BackwardBranching'
            )
            no_match = snapshot.get_attribute(
                'no_match', 'Extended/Heritage/BackwardBranching/Header'
            )
            sid_of_prog = sid - 1

            # Iterate through subhalos in the current snapshot and add
            # corresponding Subhalo objects to ´subhalos´:
            subhalos = np.empty(progenitors.size, dtype=object)
            for idx, prog_idx in enumerate(progenitors):

                # If the subhalo has a progenitor, add it to the Subhalo object of
                # the progenitor, and add to ´subhalos´ a pointer to that object:
                if prog_idx != no_match:
                    subhalos[idx] = sub_dict[sid_of_prog][prog_idx]
                    sub_dict[sid_of_prog][prog_idx].add_snapshot(idx, sid)

                # Otherwise, the subhalo has just formed, so we create a new
                # Subhalo object:
                else:
                    subhalos[idx] = Subhalo(self, idx, sid)

            sub_dict[sid] = subhalos

        return sub_dict

