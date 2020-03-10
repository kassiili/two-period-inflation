import numpy as np
import h5py
from collections import deque

from snapshot_obj import Snapshot

def trace_halo(snap_init,gn,sgn,stop=101):
    """ Traces a halo as far back in time as possible, starting from
    the given snapshot.

    Parameters
    ----------
    snapshot : Dataset object
        Starting point for the tracing.
    gn : int
        Group number of the traced halo in the initial snapshot.
    sgn : int
        Subgroup number of the traced halo in the initial snapshot.
    stop : int, optional
        Earliest snapshot to be explored

    Returns
    -------
    tracer : collections.deque object of tuples of type (float,int,int)
        Doubly linked list tracing the gn and sgn values of the halo
        through snapshots. The corresponding redshifts are included as 
        the first element of the tuples.
    """

    with h5py.File(snap_init.grp_file,'r') as grpf:
        z_init = grpf['link0/Header'].attrs.get('Redshift')

    fnum_init = snap_init.file_of_halo(gn,sgn)

    # Initialize tracer:
    tracer = deque([(z_init,gn,sgn)])

    fnum = fnum_init
    sim_ID = snap_init.simID
    snapID = snap_init.snapID
    snap = snap_init

    while snapID > stop:
        snap_prev = Snapshot(simID,snapID-1)
        gn, sgn = find_match(snap_prev, snap, gn, sgn)

    return tracer
    
def find_match(snap, snap_ref, gn, sgn):
    """ Looks for a matching halo in snap for a given halo in snap_ref.

    Parameters
    ----------
    snap : Dataset object
        Explored snapshot.
    snap_ref : Dataset object
        Reference snapshot.
    gn : int
        Group number of a halo in the reference snapshot.
    sgn : int
        Subgroup number of a halo in the reference snapshot.

    Returns
    -------
    (gn,sgn) : tuple
    """

    # Get particle IDs of reference:
    offset_ref = get_subhalo(snap_ref,'SubOffset',gn,sgn)
    length_ref = get_subhalo(snap_ref,'SubLength',gn,sgn)

    with h5py.File(snap_ref.grp_file,'r') as grpf:
        grpf['link{}/'


    return None

def get_subhalo_IDs(snap,gn,sgn):

    fnum = snap.file_of_halo(gn,sgn)
    gns = snap.get_subhalos(\
            'GroupNumber',divided=False,fnums=[fnum])[0]
    sgns = snap.get_subhalos(\
            'SubGroupNumber',divided=False,fnums=[fnum])[0]
    data = snap.get_subhalos(attr,divided=False,fnums=[fnum])[0]

    out = data[np.logical_and(gns == gn, sgns == sgn)]

    return out

def get_subhalo(snap,attr,gn,sgn):

    fnum = snap.file_of_halo(gn,sgn)
    gns = snap.get_subhalos(\
            'GroupNumber',divided=False,fnums=[fnum])[0]
    sgns = snap.get_subhalos(\
            'SubGroupNumber',divided=False,fnums=[fnum])[0]
    data = snap.get_subhalos(attr,divided=False,fnums=[fnum])[0]

    out = data[np.logical_and(gns == gn, sgns == sgn)]

    return out
