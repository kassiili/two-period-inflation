from datetime import date

import h5py
import numpy as np
from astropy import units

import datafile_oper
import dataset_comp
import match_halo


def find_subhalo_descendants(merger_tree, snap_id, index_of_subhalo):
    tracer = merger_tree.get_all_matches()

    # Get descendants from reference snapshot:
    descendant_idxs = []
    idx = index_of_subhalo
    while idx != merger_tree.no_match:
        descendant_idxs.append(idx)
        if 'Descendants' not in tracer[snap_id].keys():
            break
        idx = tracer[snap_id]['Descendants'][idx]
        snap_id += 1

    return descendant_idxs


def find_subhalo_progenitors(merger_tree, snap_id, index_of_subhalo):
    tracer = merger_tree.get_all_matches()

    # Get descendants from reference snapshot:
    progenitor_idxs = []
    idx = index_of_subhalo
    while idx != merger_tree.no_match:
        progenitor_idxs.append(idx)
        if 'Progenitors' not in tracer[snap_id].keys():
            break
        idx = tracer[snap_id]['Progenitors'][idx, 0]
        snap_id -= 1

    return progenitor_idxs


class SnapshotTracer:
    """ Traces the subhalos in a single snapshot forward and backward
    in time.

    Parameters
    ----------
    merger_tree : MergerTree object
        Contains the information about the subhalo relations between
        snapshots, based on which the tracing is done.
    no_match : int
        Value used to indicate that no match is found.
    snap_id : int
        Identifier of the snapshot that is used as the starting point of the
        tracing.
    tracer_array : ndarray of int
        Contains the index places of descendants and progenitors of the
        subhalos in snapshot snap_id.

    Notes
    -----
    When tracing in the direction, where the merger tree allows
    branching, the family lines of the most massive relatives are
    followed. Thus, a subhalo at index place j in snapshot s, with
    s > snap_id (s < snap_id), is the descendant (progenitor) of a subhalo at
    index place i in snapshot snap_id, if j == tracer_array[i, s].
    """

    def __init__(self, snap_id, merger_tree):
        self.merger_tree = merger_tree
        self.no_match = merger_tree.no_match
        self.snap_id = snap_id
        n_halos = merger_tree.simulation.get_snapshot(
            snap_id).get_subhalo_number()
        n_snaps = merger_tree.simulation.get_snap_num()
        self.tracer_array = merger_tree.no_match * \
                            np.ones((n_halos, n_snaps), dtype=int)
        self.tracer_array[:, snap_id] = np.arange(n_halos)
        self.traced_snaps = [snap_id, snap_id + 1]

    def trace(self, start=None, stop=None):

        n_snaps = np.size(self.tracer_array, axis=1)

        if start is None and stop is None:
            start = 0
            stop = n_snaps
        elif start is None or stop is None:
            if start is None:
                lim = stop
            else:
                lim = start
            start = min(self.snap_id, lim)
            stop = max(self.snap_id + 1, lim)

        self.trace_forward(stop)
        self.trace_backward(start)

        out = self.tracer_array[:, start:stop]

        return out

    def trace_forward(self, stop):

        n_halos = np.size(self.tracer_array, axis=0)

        # Set starting point to last untracked snapshot:
        start = self.traced_snaps[1] - 1
        heritage = self.merger_tree.get_all_matches()

        for sid in range(start + 1, stop):
            # Indices of the traced subhalos in the previous snapshot:
            idx_in_prev = self.tracer_array[:, sid - 1]

            # Indices of subhalos in the previous snapshot, in the
            # current snapshot:
            idx_from_prev = heritage[sid - 1]
            idx_from_prev = idx_from_prev['Descendants']

            # To avoid tracer out of bounds errors,
            # fill insufficiently short tracer arrays with no_match:
            if idx_from_prev.size < n_halos:
                fill = self.no_match * \
                       np.ones(n_halos - idx_from_prev.size)
                idx_from_prev = np.append(idx_from_prev, fill)

            # To avoid out of bounds errors, replace values that are
            # equal to no_match:
            help_idx = np.where(idx_in_prev != self.no_match,
                                idx_in_prev, np.arange(n_halos))

            idx_from_start = idx_from_prev[help_idx]

            # Save new indices of subhalos that were traced up to
            # the previous snapshot:
            self.tracer_array[:, sid] = np.where(
                idx_in_prev != self.no_match, idx_from_start,
                self.no_match)

        self.traced_snaps[1] = max(self.traced_snaps[1], stop)

    def trace_backward(self, start):

        n_halos = np.size(self.tracer_array, axis=0)

        # Set stopping point to first tracked snapshot:
        stop = self.traced_snaps[0]
        heritage = self.merger_tree.get_all_matches()

        for sid in range(stop - 1, start - 1, -1):
            # Indices of the traced subhalos in the following snapshot:
            idx_in_foll = self.tracer_array[:, sid + 1]

            # Indices of subhalos in the following snapshot, in the
            # current snapshot:
            idx_from_foll = heritage[sid + 1]
            idx_from_foll = idx_from_foll['Progenitors'][:, 0]

            # To avoid tracer out of bounds errors,
            # fill insufficiently short tracer arrays with no_match:
            if idx_from_foll.size < n_halos:
                fill = self.no_match * \
                       np.ones(n_halos - idx_from_foll.size)
                idx_from_foll = np.append(idx_from_foll, fill)

            # To avoid out of bounds errors, replace values that are
            # equal to no_match:
            help_idx = np.where(idx_in_foll != self.no_match,
                                idx_in_foll, np.arange(n_halos))

            idx_from_start = idx_from_foll[help_idx]

            # Save new indices of subhalos that were traced up to
            # the previous snapshot:
            self.tracer_array[:, sid] = np.where(
                idx_in_foll != self.no_match, idx_from_start,
                self.no_match)

        self.traced_snaps[0] = min(self.traced_snaps[0], start)

    def get_indices_at_snapshot(self, snap_id):
        return self.tracer_array[:, snap_id]

    def find_creation(self):
        creation_snaps = -1 * np.ones(np.size(self.tracer_array, axis=0))
        self.trace_backward(0)

        for snap_id in range(self.snap_id + 1):
            # Get subhalos that existed at snap and did not exist at an
            # earlier snap:
            mask_created_at_snap = np.logical_and(
                self.tracer_array[:, snap_id] != self.no_match,
                creation_snaps == -1
            )
            creation_snaps = np.where(mask_created_at_snap, snap_id,
                                      creation_snaps)

        return creation_snaps

# def creation_times(mtree, snap_start, snap_stop):
#     snap_ids = np.arange(snap_stop, snap_start, -1) - 1
#     created = {}
#
#     for sid in snap_ids:
#
#         snap = mtree.simulation.snapshots[sid]
#         created[sid] = np.full(snap.get_subhalo_number(), sid)
#
#         if sid == snap_start:
#             break
#
#         prog = snap.get_subhalos('Progenitors', h5_group='Extended/Heritage')
#
#         # Masking array for subhalos that have a progenitor:
#         mask_has_prog = prog != mtree.no_match
#
#         # Loop through subsequent snapshots:
#         for snap_created in filter(
#             lambda item: item[0] > sid, created.items()
#         ):
#             mask = np.logical_and(snap_created == sid, mask_has_prog)
#             snap_created[mask] = sid
#
# def creation_and_destruction_times(mtree, snap_start, snap_stop):
#     snap_ids = np.arange(snap_start, snap_stop)
#     created = {}
#     destroyed = {}
#
#     for sid in snap_ids:
#
#         snap = mtree.simulation.snapshots[sid]
#         created[sid] = np.full(snap.get_subhalo_number(), sid)
#         destroyed[sid] = np.full(snap.get_subhalo_number(), sid)
#
#         if sid == snap_start:
#             continue
#
#         prog = snap.get_subhalos('Progenitors', h5_group='Extended/Heritage')
#         sid_prev = sid - 1
#
#         # Masking array for subhalos that have a progenitor:
#         mask_prog = prog[:,0] != mtree.no_match
#
#         # Get pointers of the progenitors (for subhalos with no progenitors,
#         # insert dummy elements (of value None) to get the correct shape):
#         pointer_of_prog = subhalo_pointers[sid_prev][
#             np.where(mask_prog, prog[:,0], None)
#         ]
#
#         # If the subhalo has no progenitor, create a new pointer, otherwise use
#         # the pointer of the (most massive) progenitor:
#         subhalo_pointers[sid] = np.where(
#             mask_prog,
#             pointer_of_prog,
#             ["Snap{}_GN{}_SGN{}".format(sid, gn, sgn)
#              for gn, sgn in zip(gns, sgns)]
#         )
#
#         # Add indices in current snapshot to subhalos present in the previous
#         # snapshot:
#         for i, pointer in zip(prog[mask_prog,0], pointer_of_prog[mask_prog]):
#             subhalos[pointer].append((sid, i))
#
#         # Add items for new subhalos:
#         mask_no_prog = np.logical_not(mask_prog)
#         subhalos.update({
#             "Snap{}_GN{}_SGN{}".format(sid, gn, sgn) : [(sid, i)]
#             for i, gn, sgn in zip(
#                 np.arange(gns.size)[mask_no_prog],
#                 gns[mask_no_prog], sgns[mask_no_prog]
#             )
#         })
#
#     # Sort snapshot index lists of each subhalo by snapshot IDs (in ascending
#     # order in time):
#     for sub_id, sub in subhalos.items():
#         subhalos[sub_id] = sorted(sub, key=lambda in_snap: in_snap[0])


def creation_times(sub_dict):

    created = {}

    for sid, subhalos in sub_dict.items():
        created[sid] = np.array([
            min(sub.get_indices()[1]) for sub in subhalos
        ])

    return created


def destruction_times(sub_dict):

    destroyed = {}

    for sid, subhalos in sub_dict.items():
        destroyed[sid] = np.array([
            max(sub.get_indices()[1]) for sub in subhalos
        ])

    return destroyed

# NOTE: satellites that fly out (stop being satellites, but are not
# destroyed) are handled the same as satellites that get destroyed.
# Would it be better to ignore such fly-by subhalos?
def get_fallin_times_old(sub_dict, simulation, central, sat_func=None):

    if not sat_func:
        def sat_func(snapshot, central):
            r200 = snapshot.get_subhalos(
                'Group_R_TopHat200', h5_group='FOF'
            )[central[0]] * units.cm.to(units.kpc)
            return dataset_comp.satellites_of_halo_by_distance(
                snapshot, central, max_dist=r200
            )

    fallin_times = {}

    # Iterate through snapshots in ascending order:
    snap_ids = list(sub_dict.keys())
    sorting = np.argsort(snap_ids)
    snap_ids = np.array(snap_ids)[sorting]
    snap_subs = np.array(list(sub_dict.values()))[sorting]
    for sid, subhalos in zip(snap_ids, snap_subs):

        # Get satellite masking array:
        snapshot = simulation.get_snapshot(sid)
        mask_sat = sat_func(snapshot, central)

        fallin_times[sid] = np.full(mask_sat.size, None)

        # Set fall-in times of all satellites in the first snapshot at that
        # snapshot:
        if sid == min(snap_ids):
            fallin_times[sid][mask_sat] = sid
            mask_sat_prev = mask_sat
            continue

        # Iterate over satellites:
        for idx, sat in zip(np.arange(mask_sat.size)[mask_sat],
                            subhalos[mask_sat]):

            # If satellite is not present in the previous snapshot,
            # set fall-in time to ´sid´:
            idx_prev = sat.get_index_at_snap(sid-1)
            if not idx_prev:
                fallin_times[sid][idx] = sid
            else:
                # If it had already fallen in, at the previous snapshot,
                # copy fall-in time from previous:
                if mask_sat_prev[idx_prev]:
                    fallin_times[sid][idx] = fallin_times[sid-1][idx_prev]
                else:
                    fallin_times[sid][idx] = sid

        mask_sat_prev = mask_sat

    return fallin_times

# NOTE: satellites that fly out (stop being satellites, but are not
# destroyed) are handled the same as satellites that get destroyed.
# Would it be better to ignore such fly-by subhalos?
def get_fallin_times(simulation, central, snap_start=None, snap_stop=None,
                     sat_func=None):

    if not sat_func:
        def sat_func(snapshot, central):
            r200 = central.get_fof_data(
                'Group_R_TopHat200', snapshot.snap_id
            )[0] * units.cm.to(units.kpc)
            return dataset_comp.satellites_of_halo_by_distance(
                snapshot, central, max_dist=r200
            )

    if not snap_start or not snap_stop:
        snap_ids = simulation.get_snap_ids()
    else:
        snap_ids = np.arange(snap_start, snap_stop)

    fallin_times = {}

    # Iterate through snapshots in ascending order:
    for sid in snap_ids:

        # Get satellite masking array:
        snapshot = simulation.get_snapshot(sid)
        mask_sat = sat_func(snapshot, central)
        # print(snapshot.get_subhalo_number(), mask_sat.size)

        fallin_times[sid] = np.full(snapshot.get_subhalo_number(), np.nan)

        # Set fall-in times of all satellites in the first snapshot at that
        # snapshot:
        if sid == min(snap_ids):
            # print('here', sid)
            fallin_times[sid][mask_sat] = sid
            mask_sat_prev = mask_sat
            continue

        prog = snapshot.get_subhalos('Progenitors',
                                     'Extended/Heritage/BackwardBranching')
        no_match = snapshot.get_attribute(
            'no_match', 'Extended/Heritage/BackwardBranching/Header'
        )

        # Fill no_match values with dummy index:
        mask_has_prog = (prog != no_match)
        # print(sid, max(prog[mask_has_prog]), mask_sat_prev.size)
        prog_wdummy = np.where(mask_has_prog, prog, 0)

        mask_prog_is_sat = np.logical_and(
            mask_has_prog, mask_sat_prev[prog_wdummy]
        )

        fallin_times[sid] = np.where(
            mask_sat,
            np.where(
                mask_prog_is_sat,
                fallin_times[sid - 1][prog_wdummy],
                sid
            ),
            np.nan
        )

        mask_sat_prev = mask_sat

    return fallin_times


def get_fallin_times_lg(simulation, m31, mw, snap_start=None, snap_stop=None,
                        sat_func=None, first_infall=True):
    """ Find the fall-in times of satellites of M31 and MW in each snapshot.

    Parameters
    ----------
    simulation : Simulation object
    m31 : Subhalo object
    mw : Subhalo object
    first_infall : bool
        If True (default), then the time a subhalo is first identified as a
        satellite is copied as the infall time for all the following
        snapshots, even if the subhalo flies out of the central galaxy
        afterwards. If False, then only the latest infall time is copied for
        only those following snapshots, at which the subhalo is a satellite.

    Returns
    -------
    fallin_times_m31, fallin_times_mw : dict of ndarray of int
        Key - snapshot ID, Value - array of length SubNum of that snapshot,
        with each item indicating the fall-in snapshot ID of that subhalo,
        if the subhalo is a satellite, otherwise the item is np.nan.
    """
    if first_infall:
        return get_fallin_times_lg_first_infall(
            simulation, m31, mw, snap_start=snap_start, snap_stop=snap_stop,
            sat_func=sat_func
        )
    else:
        return get_fallin_times_lg_last_infall(
            simulation, m31, mw, snap_start=snap_start, snap_stop=snap_stop,
            sat_func=sat_func
        )

def get_fallin_times_lg_last_infall(simulation, m31, mw, snap_start=None,
                                    snap_stop=None, sat_func=None):
    """ Find the fall-in times of satellites of M31 and MW in each snapshot.

    Parameters
    ----------
    simulation : Simulation object
    m31 : Subhalo object
    mw : Subhalo object

    Returns
    -------
    fallin_times_m31, fallin_times_mw : dict of ndarray of int
        Key - snapshot ID, Value - array of length SubNum of that snapshot,
        with each item indicating the fall-in snapshot ID of that subhalo,
        if the subhalo is a satellite, otherwise the item is np.nan.

    Notes
    -----
    The fall-in time of any given subhalo will only be written in dictionary
    items corresponding to snapshots that are temporally after the fall-in.
    Similarly, if a satellite flies out of the central, then no fall-in time
    will be recorded after it is no longer seen as a satellite. If it
    afterwards falls back in, the fall-in time that is recorded from there
    on is the time of the second fall-in.
    """

    if not sat_func:
        def sat_func(snapshot, halo1, halo2):
            h1_id = halo1.get_group_number_at_snap(snapshot.snap_id)
            h2_id = halo2.get_group_number_at_snap(snapshot.snap_id)
            mask_sat1, mask_sat2,_ = dataset_comp.split_satellites_by_distance(
                snapshot, h1_id, h2_id, sat_r=300, comov=True
            )
            return mask_sat1, mask_sat2

    if not snap_start or not snap_stop:
        snap_ids = simulation.get_snap_ids()
    else:
        snap_ids = np.arange(snap_start, snap_stop)

    fallin_times_m31 = {}
    fallin_times_mw = {}

    # Iterate through snapshots in ascending order:
    for sid in snap_ids:

        # Get satellite masking array:
        snapshot = simulation.get_snapshot(sid)
        mask_m31_sat, mask_mw_sat = sat_func(snapshot, m31, mw)
        # print(snapshot.get_subhalo_number(), mask_sat.size)

        fallin_times_m31[sid] = np.full(snapshot.get_subhalo_number(), np.nan)
        fallin_times_mw[sid] = np.full(snapshot.get_subhalo_number(), np.nan)

        # Set fall-in times of all satellites in the first snapshot at that
        # snapshot:
        if sid == min(snap_ids):
            # print('here', sid)
            fallin_times_m31[sid][mask_m31_sat] = sid
            fallin_times_mw[sid][mask_mw_sat] = sid
            mask_m31_sat_prev = mask_m31_sat
            mask_mw_sat_prev = mask_mw_sat
            continue

        prog = snapshot.get_subhalos('Progenitors',
                                     'Extended/Heritage/BackwardBranching')
        no_match = snapshot.get_attribute(
            'no_match', 'Extended/Heritage/BackwardBranching/Header'
        )

        # Fill no_match values with dummy index:
        mask_has_prog = (prog != no_match)
        # print(sid, max(prog[mask_has_prog]), mask_sat_prev.size)
        prog_wdummy = np.where(mask_has_prog, prog, 0)

        mask_prog_is_m31_sat = np.logical_and(
            mask_has_prog, mask_m31_sat_prev[prog_wdummy]
        )
        mask_prog_is_mw_sat = np.logical_and(
            mask_has_prog, mask_mw_sat_prev[prog_wdummy]
        )

        fallin_times_m31[sid] = np.where(
            mask_m31_sat,
            np.where(
                mask_prog_is_m31_sat,
                fallin_times_m31[sid - 1][prog_wdummy],
                sid
            ),
            np.nan
        )

        fallin_times_mw[sid] = np.where(
            mask_mw_sat,
            np.where(
                mask_prog_is_mw_sat,
                fallin_times_mw[sid - 1][prog_wdummy],
                sid
            ),
            np.nan
        )

        mask_m31_sat_prev = mask_m31_sat
        mask_mw_sat_prev = mask_mw_sat

    return fallin_times_m31, fallin_times_mw


def get_fallin_times_lg_first_infall(simulation, m31, mw, snap_start=None,
                                     snap_stop=None, sat_func=None):
    """ Find the fall-in times of satellites of M31 and MW in each snapshot.

    Parameters
    ----------
    simulation : Simulation object
    m31 : Subhalo object
    mw : Subhalo object

    Returns
    -------
    fallin_times_m31, fallin_times_mw : dict of ndarray of int
        Key - snapshot ID, Value - array of length SubNum of that snapshot,
        with each item indicating the fall-in snapshot ID of that subhalo,
        if the subhalo is a satellite, otherwise the item is np.nan.
    """

    if not sat_func:
        def sat_func(snapshot, halo1, halo2):
            h1_id = halo1.get_group_number_at_snap(snapshot.snap_id)
            h2_id = halo2.get_group_number_at_snap(snapshot.snap_id)
            mask_sat1, mask_sat2,_ = dataset_comp.split_satellites_by_distance(
                snapshot, h1_id, h2_id, sat_r=300, comov=True
            )
            return mask_sat1, mask_sat2

    if not snap_start or not snap_stop:
        snap_ids = simulation.get_snap_ids()
    else:
        snap_ids = np.arange(snap_start, snap_stop)

    fallin_times_m31 = {}
    fallin_times_mw = {}

    # Iterate through snapshots in ascending order:
    for sid in snap_ids:

        # Get satellite masking array:
        snapshot = simulation.get_snapshot(sid)
        mask_m31_sat, mask_mw_sat = sat_func(snapshot, m31, mw)

        # Set fall-in times of all satellites in the first snapshot at that
        # snapshot:
        if sid == min(snap_ids):
            fallin_times_m31[sid] = np.where(mask_m31_sat, sid, np.nan)
            fallin_times_mw[sid] = np.where(mask_mw_sat, sid, np.nan)
            continue

        # Selection for subhalos with progenitors:
        prog = snapshot.get_subhalos('Progenitors',
                                     'Extended/Heritage/BackwardBranching')
        no_match = snapshot.get_attribute(
            'no_match', 'Extended/Heritage/BackwardBranching/Header'
        )
        mask_has_prog = (prog != no_match)

        # Fill no_match values with dummy index:
        prog_wdummy = np.where(mask_has_prog, prog, 0)

        # Get infall times of progenitors:
        prog_fallin_time_m31 = np.where(
            mask_has_prog,
            fallin_times_m31[sid - 1][prog_wdummy],
            np.nan
        )
        # Copy progenitor infall times, where listed. Add infall times for
        # new satellites:
        fallin_times_m31[sid] = np.where(
            ~np.isnan(prog_fallin_time_m31),
            prog_fallin_time_m31,
            np.where(mask_m31_sat, sid, np.nan)
        )

        prog_fallin_time_mw = np.where(
            mask_has_prog,
            fallin_times_mw[sid - 1][prog_wdummy],
            np.nan
        )
        fallin_times_mw[sid] = np.where(
            ~np.isnan(prog_fallin_time_mw),
            prog_fallin_time_mw,
            np.where(mask_mw_sat, sid, np.nan)
        )

    return fallin_times_m31, fallin_times_mw

def sf_onset_times(simulation, snap_start=None, snap_stop=None):
    """ Find the star formation onset times for all subhalos.

    Parameters
    ----------
    simulation : Simulation object

    Returns
    -------
    sf_onset : dict of ndarray of int
        Key - snapshot ID, Value - array of length SubNum of that snapshot,
        with each item indicating the SF onset time snapshot ID of that subhalo,
        if the subhalo has any stars, otherwise the item is np.nan.

    Notes
    -----
    """

    if not snap_start or not snap_stop:
        snap_ids = simulation.get_snap_ids()
    else:
        snap_ids = np.arange(snap_start, snap_stop)

    sf_onset = {}

    # Iterate through snapshots in ascending order:
    for sid in snap_ids:

        # Get satellite masking array:
        snapshot = simulation.get_snapshot(sid)
        mask_lum, _ = dataset_comp.split_luminous(snapshot)

        sf_onset[sid] = np.full(snapshot.get_subhalo_number(), np.nan)

        # Set fall-in times of all satellites in the first snapshot at that
        # snapshot:
        if sid == min(snap_ids):
            sf_onset[sid][mask_lum] = sid
            mask_lum_prev = mask_lum
            continue

        prog = snapshot.get_subhalos('Progenitors',
                                     'Extended/Heritage/BackwardBranching')
        no_match = snapshot.get_attribute(
            'no_match', 'Extended/Heritage/BackwardBranching/Header'
        )

        # Fill no_match values with dummy index:
        mask_has_prog = (prog != no_match)
        prog_wdummy = np.where(mask_has_prog, prog, 0)

        mask_prog_is_lum = np.logical_and(
            mask_has_prog, mask_lum_prev[prog_wdummy]
        )

        sf_onset[sid] = np.where(
            mask_lum,
            # If subhalo is luminous and has a luminous progenitor, then set its
            # sf onset time equal to that of the progenitor:
            np.where(
                mask_prog_is_lum,
                sf_onset[sid - 1][prog_wdummy],
                sid
            ),
            np.nan
        )

        mask_lum_prev = mask_lum

    return sf_onset


def get_all_satellites_as_array(simulation, m31_ref, mw_ref, snap_ref,
                                snap_start, snap_stop):
    """ Find all subhalos that are satellites of M31 or MW between the given
    snapshots.

    Returns
    -------
    m31_sats, mw_sats : set
        Sets of Subhalo objects of the satellites of both centrals.
    """

    sub_dict = trace_subhalos(simulation, snap_start, snap_stop)
    m31 = sub_dict[snap_ref][
        simulation.get_snapshot(snap_ref).index_of_halo(m31_ref[0], m31_ref[1])
    ]
    mw = sub_dict[snap_ref][
        simulation.get_snapshot(snap_ref).index_of_halo(mw_ref[0], mw_ref[1])
    ]

    m31_sats_set = set()
    mw_sats_set = set()

    m31_sats = []
    mw_sats = []
    m31_idx_fallin = []
    m31_snap_fallin = []
    m31_vmax_fallin = []
    mw_idx_fallin = []
    mw_snap_fallin = []
    mw_vmax_fallin = []

    for snap_id in range(snap_start, snap_stop):
        snap = simulation.get_snapshot(snap_id)
        vmax = snap.get_subhalos('Max_Vcirc', 'Extended')[:,0]
        mask_m31, mask_mw,_ = dataset_comp.split_satellites_whack(
            snap, m31, mw, sat_r=300, comov=True
        )

        # Get new satellites in a list:
        new_m31 = [sat for sat in sub_dict[snap_id][mask_m31]
                   if sat not in m31_sats_set]
        new_mw = [sat for sat in sub_dict[snap_id][mask_mw]
                  if sat not in mw_sats_set]

        m31_sats.extend(new_m31)
        mw_sats.extend(new_mw)

        # Save the fallin times and indices at fallin for the new satellites:
        m31_new_inds = [sat.get_index_at_snap(snap_id) for sat in new_m31]
        m31_idx_fallin.extend(m31_new_inds)
        m31_snap_fallin.extend([snap_id] * len(new_m31))
        m31_vmax_fallin.extend(vmax[m31_new_inds])

        mw_new_inds = [sat.get_index_at_snap(snap_id) for sat in new_mw]
        mw_idx_fallin.extend(mw_new_inds)
        mw_snap_fallin.extend([snap_id] * len(new_mw))
        mw_vmax_fallin.extend(vmax[mw_new_inds])

        # Add the new satellites into the sets:
        m31_sats_set.update(sub_dict[snap_id][mask_m31])
        mw_sats_set.update(sub_dict[snap_id][mask_mw])

    # Sort by vmax:
    m31_sorting = np.argsort(-m31_vmax_fallin)
    mw_sorting = np.argsort(-mw_vmax_fallin)

    return (m31_sats[m31_sorting], m31_snap_fallin[m31_sorting]), \
           (mw_sats[mw_sorting], mw_snap_fallin[mw_sorting])


def get_all_satellites(simulation, m31_ref, mw_ref, snap_ref, snap_start,
                       snap_stop):
    """ Find all subhalos that are satellites of M31 or MW between the given
    snapshots.

    Returns
    -------
    m31_sats, mw_sats : set
        Sets of Subhalo objects of the satellites of both centrals.
    """

    sub_dict = trace_subhalos(simulation, snap_start, snap_stop)
    m31 = sub_dict[snap_ref][
        simulation.get_snapshot(snap_ref).index_of_halo(m31_ref[0], m31_ref[1])
    ]
    mw = sub_dict[snap_ref][
        simulation.get_snapshot(snap_ref).index_of_halo(mw_ref[0], mw_ref[1])
    ]

    m31_sats = set()
    mw_sats = set()

    for snap_id in range(snap_start, snap_stop):
        snap = simulation.get_snapshot(snap_id)
        mask_m31, mask_mw,_ = dataset_comp.split_satellites_whack(
            snap, m31, mw, sat_r=300, comov=True
        )

        # Add the new satellites into the sets:
        m31_sats.update(sub_dict[snap_id][mask_m31])
        mw_sats.update(sub_dict[snap_id][mask_mw])

    return m31_sats, mw_sats


def order_satellites_by_mass_at_fallin(simulation, sats, central):
    return None


class MergerTree:
    """ Encapsulates the relations between subhalos in different snapshots.

    Parameters
    ----------
    simulation : Simulation object
        The simulation, whose merger tree this object represents.
    matcher : SnapshotMatcher object
        The object used, whenever links between subhalos in two snapshots
         are searched.
    no_match : int
        Value used to indicate that no match is found.
    branching : str
        Either 'BackwardBranching' or 'ForwardBranching'. In the former case,
        subhalo mergers are identified but fragmentation ignored, and vice
        versa, in the latter case.
    min_snaps_traced : int
        NOT IMPLEMENTED!
        The minimum number of snapshots that any subhalo must be traced over
        in order for it to be treated as non-volatile. The connections
        between snapshots of a volatile subhalo are ignored.
    h5_group : str
        Name of the HDF5 group, under which the merger tree information is
        stored in each snapshot.
    storage_file : str
        File name of a backup storage file of the merger tree.

    Notes
    -----
    The merger tree is implemented by two sets of NumPy arrays (of type int):
    two arrays called descendants and progenitors are defined for each
    snapshot in the simulation. The descendants array is defined by the
    following statement:

    Let a subhalo at index place i in snapshot s be identified (by matcher)
    with a subhalo at index place j in snapshot s+1. Then:
    j == descendants[i] is True.

    Similarly, for the progenitors. These arrays are stored in the HDF5 files
    of the corresponding snapshot.

    Handling volatile subhalos (with min_snaps_traced) is not crucially
    necessary, if the minimum number of subhalo member particles is
    sufficiently high.
    """

    def __init__(self, simulation, matcher=None,
                 branching='BackwardBranching',
                 min_snaps_traced=1):
        """

        Parameters
        ----------
        branching : str
            The merger tree only allows branching in one direction (in
            time), which is indicated by this value.
        """
        self.simulation = simulation
        if matcher is None:
            # Use the default matcher, which identifies subhalos one-to-one:
            self.matcher = match_halo.SnapshotMatcher()
        else:
            self.matcher = matcher
        self.no_match = self.matcher.no_match
        self.branching = branching
        self.min_snaps_traced = min_snaps_traced
        self.h5_group = 'Extended/Heritage/{}'.format(self.branching)
        self.storage_file = '.tracer_{}_{}.hdf5'.format(
            self.branching, self.simulation.sim_id)
        self.create_header()

    def create_header(self):
        """ Create a header group, in each snapshot, for the values that define
        the linking conditions. """
        for snapshot in self.simulation.get_snapshots():
            with h5py.File(snapshot.group_data.fname, 'r+') as f:
                header = f.require_group("{}/Header".format(self.h5_group))
                header.attrs.create("no_match", self.no_match)
                header.attrs.create("n_matches", self.matcher.n_matches)
                header.attrs.create("f_link_srch", self.matcher.f_link_srch)
                header.attrs.create("f_mass_link", self.matcher.f_mass_link)
                header.attrs.create("n_link_ref", self.matcher.n_link_ref)
                header.attrs.create("CreationDate",
                                    date.today().strftime("%d/%m/%Y"))

    def build_tree(self, snap_id_1, snap_id_2, overwrite=False):
        """ Find descendants and progenitors of all subhalos between
        given snapshots.

        Parameters
        ----------
        overwrite : bool
            If True, then existing merger tree data is overwritten, otherwise
            just read.
        """

        if self.branching == 'ForwardBranching':
            self.build_tree_with_forward_branch(snap_id_1, snap_id_2,
                                                overwrite)
        else:
            self.build_tree_with_back_branch(snap_id_1, snap_id_2, overwrite)

    def build_tree_with_back_branch(self, snap_id_1, snap_id_2, overwrite):
        """ Find subhalo heritage iterating forward in time.

        Notes
        -----
        In this case, when two subsequent snapshots are compared,
        the subhalos in the earlier snapshot will only be matched with
        a single subhalo in the next snapshot, so that branching only
        happens 'backward' in time. However, if ´self.matcher.n_matches'==1,
        then subhalos are linked one-to-one.
        """

        snap_start = min(snap_id_1, snap_id_2)
        snap_stop = max(snap_id_1, snap_id_2)

        # Get the first snapshot:
        snap = self.simulation.get_snapshot(snap_start)

        while snap.snap_id < snap_stop - 1:
            # Get next snapshot for matching:
            snap_next = self.get_next_snap(snap.snap_id)
            if snap_next is None:
                break

            snap_desc_exists = snap.group_data.dataset_exists('Descendants',
                                                              self.h5_group)
            snap_next_prog_exists = \
                snap_next.group_data.dataset_exists('Progenitors',
                                                    self.h5_group)

            # If asked to overwrite, or one of the datasets is missing,
            # do the matching, and save:
            if overwrite or not snap_desc_exists or not snap_next_prog_exists:
                str_unf = "Matching snapshot {} with snapshot {} " + \
                         "(in simulation {})"
                print(str_unf.format(snap.snap_id, snap_next.snap_id,
                                     self.simulation.sim_id))
                snap_desc, snap_next_prog = \
                    self.matcher.match_snapshots(snap, snap_next)

                # Save matches to the subhalo catalogues:
                snap.group_data.save_dataset(
                    snap_desc, 'Descendants', self.h5_group
                )
                snap_next.group_data.save_dataset(
                    snap_next_prog, 'Progenitors', self.h5_group
                )

            snap = snap_next

        # Remove connections of volatile subhalos:
        if self.min_snaps_traced > 1:
            self.prune_tree() # NOT IMPLEMENTED

    # NOT VERIFIED!
    def build_tree_with_forward_branch(self, snap_id_1, snap_id_2, overwrite):

        snap_start = max(snap_id_1, snap_id_2)
        snap_stop = min(snap_id_1, snap_id_2)

        # Get the first snapshot:
        snap = self.simulation.get_snapshot(snap_start)

        while snap.snap_id != snap_stop - 1:
            # Get next snapshot for matching:
            snap_next = self.get_next_snap(snap.snap_id)
            if snap_next is None:
                break

            # If matches are already saved, read them - otherwise, do the
            # matching:
            progenitors = snap.get_subhalos('Progenitors', self.h5_group)
            descendants_next = snap_next.get_subhalos('Descendants',
                                                      self.h5_group)

            if descendants_next.size == 0 or progenitors.size == 0:
                descendants_next, progenitors = \
                    self.matcher.match_snapshots(snap, snap_next)
                # Save matches to the subhalo catalogues:
                datafile_oper.save_dataset(
                    descendants_next, 'Descendants', self.h5_group, snap_next)
                datafile_oper.save_dataset(
                    progenitors, 'Progenitors', self.h5_group, snap)

            snap = snap_next

        # Remove connections of volatile subhalos:
        if self.min_snaps_traced > 1:
            self.prune_tree()

    def get_next_snap(self, cur_snap_id):
        # Set snap_id incrementation value:
        if self.branching == 'ForwardBranching':
            incr = -1
        else:
            incr = 1

        snap_next = self.simulation.get_snapshot(cur_snap_id + incr)

        return snap_next

    def store_tracer(self):
        """ Save another copy of the matches into a HDF5 file. """

        match_dict = self.get_all_matches()

        # Save all non-empty entries to the storage file:
        with h5py.File(self.storage_file, 'w') as f:
            for snap_id, matches in match_dict.items():
                h5_group = '/Heritage/{}/{}'.format(self.branching,
                                                    str(snap_id))
                f.create_group(h5_group)
                for key, arr in matches.items():
                    if arr.size != 0:
                        f[h5_group].create_dataset(key, data=arr)

        return match_dict

    def get_all_matches(self):
        """ Return all found matches in a dictionary. """

        # Get snapshot identifiers:
        snap_ids = self.simulation.get_snap_ids()

        # Add all non-empty entries to the output dictionary:
        match_dict = dict()
        for snap_id in snap_ids:
            snap = self.simulation.get_snapshot(snap_id)
            if snap is None:
                continue

            descendants = snap.get_subhalos('Descendants', self.h5_group)
            progenitors = snap.get_subhalos('Progenitors', self.h5_group)
            if descendants.size != 0 and progenitors.size != 0:
                match_dict[snap_id] = {'Descendants': descendants,
                                       'Progenitors': progenitors}
            elif descendants.size != 0:
                match_dict[snap_id] = {'Descendants': descendants}
            elif progenitors.size != 0:
                match_dict[snap_id] = {'Progenitors': progenitors}

        return match_dict

    def read_from_storage(self):

        match_dict = {}

        with h5py.File(self.storage_file, 'r') as f:
            for snap_id, matches in f['Heritage/{}'.format(
                    self.branching)].items():
                snap_id = int(snap_id)
                match_dict[snap_id] = {}

                snap = self.simulation.get_snapshot(snap_id)
                for key, dataset in matches.items():
                    match_dict[snap_id][key] = dataset[...]

                    # Save matches to the subhalo catalogues:
                    datafile_oper.save_dataset(
                        dataset[...], key, self.h5_group, snap)

        return match_dict

    #    def get_matched_snapshots(self):
    #
    #        # Get snapshot identifiers:
    #        snap_ids = self.simulation.get_snap_ids()
    #
    #        matched_snap_ids = set()
    #        for snap_id in snap_ids:
    #            # Get the two potentially matched snapshots:
    #            snap = self.simulation.get_snapshot(snap_id)
    #            snap_next = self.get_next_snap(snap_id)
    #            if snap is None or snap_next is None:
    #                continue
    #
    #            # Get the matches, and if they exist, add to the set of
    #            # matched:
    #            h5_group = "Extended/Matches/{}/{}".format(
    #                self.simulation.sim_id, snap_next.snap_id)
    #            matches = snap.get_subhalos("Matches", h5_group)
    #            if matches.size != 0:
    #                matched_snap_ids.add(snap_id)
    #                matched_snap_ids.add(snap_next.snap_id)
    #
    #        # Get the snapshots, ordered by identifiers:
    #        matched_snaps = np.array(
    #            [self.simulation.get_snapshot(snap_id) for snap_id in
    #             np.sort(list(matched_snap_ids))])
    #
    #        return matched_snaps

    def prune_tree(self):
        # Iterate through the snapshots, keeping track of how many
        # snapshots each subhalo instance extends over. Then remove the
        # connections of subhalos with less than a given amount of
        # connections from the merger tree.
        pass
