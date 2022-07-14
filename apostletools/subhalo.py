import numpy as np

import dataset_comp


class Subhalo:

    def __init__(self, simulation, idx=None, snap_id=None):

        self.simulation = simulation
        self.indices = set()
        if idx is not None:
            self.add_snapshot(idx, snap_id)

    def __hash__(self):
        """ Create the hash based on the host simulation and the index at
        formation time. """
        idx_form, snap_form = self.index_at_formation()
        return hash((self.simulation.sim_id, idx_form, snap_form))

    def __eq__(self, other):
        if isinstance(other, Subhalo):
            return ((self.simulation.sim_id == other.simulation.sim_id) and
                    (self.index_at_formation() == other.index_at_formation()))
        else:
            return False

    def index_at_formation(self):
        inds, snap_ids = self.get_indices()
        return inds[0], snap_ids[0]

    def index_at_destruction(self):
        inds, snap_ids = self.get_indices()
        return inds[-1], snap_ids[-1]

    def get_indices(self, return_snap_ids=True):
        """ Get the index place of the subhalo in each snapshot, where it
        appears. """
        # Get as list and sort by snap_id:
        inds = list(self.indices)
        inds = sorted(inds, key=lambda idx_el: idx_el[1])

        inds, snap_ids = map(list, zip(*inds))

        if return_snap_ids:
            return inds, snap_ids
        else:
            return inds

    def get_snap_ids(self):
        return self.get_indices(return_snap_ids=True)[1]

    def get_index_at_snap(self, snap_id):
        inds, snap_ids = self.get_indices()
        if snap_id not in snap_ids:
            return None
        else:
            idx = inds[snap_ids.index(snap_id)]
            return idx

    def get_group_number_at_snap(self, snap_id):
        return (self.get_halo_data("GroupNumber", snap_id)[0],
                self.get_halo_data("SubGroupNumber", snap_id)[0])

    def add_snapshot(self, idx, snap_id):
        self.indices.add((idx, snap_id))

    def get_halo_data(self, dset_name, snap_ids, h5_group='Subhalo'):
        """ Retrieves a subhalo dataset in the given snapshots.
        """

        try:
            iterator = iter(snap_ids)
        except TypeError:
            # If ´snap_id´ is not iterable (but, for instance, an integer):
            snap_ids = [snap_ids]

        data = np.array([
            self.simulation.get_snapshot(sid).get_subhalos(
                dset_name, h5_group=h5_group
            )[self.get_index_at_snap(sid)] for sid in snap_ids
        ])

        return data

    def get_fof_data(self, dset_name, snap_ids):
        """ Retrieves a subhalo dataset in the given snapshot.
        """

        try:
            iterator = iter(snap_ids)
        except TypeError:
            # If ´snap_id´ is not iterable (but, for instance, an integer):
            snap_ids = [snap_ids]

        # The FOF datasets are ordered by ascending group number (with index
        # 0 having gn 1):
        gns = self.get_halo_data('GroupNumber', snap_ids).astype(int)
        data = np.array([
            self.simulation.get_snapshot(sid).get_subhalos(
                dset_name, h5_group='FOF'
            )[gn - 1] for sid, gn in zip(snap_ids, gns)
        ])

        return data

    def distance_to_central(self, central, snap_ids, centre_name=None):
        """ Compute distance to the central galaxy at the given snapshot.
        """

        try:
            iterator = iter(snap_ids)
        except TypeError:
            # If ´snap_id´ is not iterable (but, for instance, an integer):
            snap_ids = [snap_ids]

        if centre_name is None:
            centre_name = "CentreOfPotential"

        halo_centre = self.get_halo_data(centre_name, snap_ids)
        central_centre = central.get_halo_data(centre_name, snap_ids)

        # Wrap around centre:
        halo_centre = np.array([
            dataset_comp.periodic_wrap(self.simulation.get_snapshot(sid),
                                       cctr, hctr)
            for sid, cctr, hctr in zip(snap_ids, central_centre, halo_centre)])

        return central_centre - halo_centre

    def distance_to_self(self, snap_ids, centre_name=None):
        """ For all subhalos in the simulation, compute distance to this
        subhalo.
        """

        try:
            iterator = iter(snap_ids)
        except TypeError:
            # If ´snap_id´ is not iterable (but, for instance, an integer):
            snap_ids = [snap_ids]

        if centre_name is None:
            centre_name = "CentreOfPotential"

        cop_self = self.get_halo_data(centre_name, snap_ids)

        # Get, in a list, arrays of cops of all subhalos at each snapshot:
        cop_all = list(
            self.simulation.get_subhalos(snap_ids, centre_name).values()
        )

        snaps = self.simulation.get_snapshots(snap_ids)
        dist = {snap.snap_id: \
                    dataset_comp.periodic_wrap(snap, centre, cop) - centre
                for snap, centre, cop in zip(snaps, cop_self, cop_all)}

        return dist
