import numpy as np

from Snapshot import snapshot_obj

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

    # Get particle IDs:

    return None


