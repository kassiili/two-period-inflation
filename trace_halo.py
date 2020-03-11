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
        gn, sgn = match_snapshots(snap_prev, snap, gn, sgn)

    return tracer
    
def match_snapshots(snap, snap_ref, gn, sgn):
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

    # Get particle IDs and halo mass of reference:
    IDs_ref, = get_subhalo_IDs(snap_ref,gn,sgn)
    mass_ref, = get_subhalo(snap_ref,'Mass',gn,sgn)

    # Get index of halo with same sgn and gn as ref:
    fnum_ref = snap_ref.file_of_halo(gn,sgn)
    gns = snap.get_subhalos('GroupNumber',False,fnums=[fnum_ref])[0]
    sgns = snap.get_subhalos('SubGroupNumber',False,fnums=[fnum_ref])[0]
    idx = np.argwhere(np.logical_and((gns==gn),(sgns==sgn)))
    print(gns[idx],sgns[idx])

    IDs_in_file = snap.get_subhalos_IDs(fnums=[fnum_ref])
    mass_in_file = snap.get_subhalos('Mass',False,fnums=[fnum_ref])[0]

    # Initial values:
    IDs = IDs_in_file[idx] 
    mass = mass_in_file[idx]
    found_match = False
    step = 0
    while not found_match:
        found_match = match_subhalos(IDs_ref,mass_ref,IDs,mass)

    return False

def match_subhalos(IDs1,mass1,IDs2,mass2):
    """ Check if two halos in different snapshots correspond to the same
    halo. 

    Parameters
    ----------
    IDs1 : ndarray of int
        Particle IDs of first halo.
    mass1 : float
        Mass of (particles of) first halo.
    IDs2 : ndarray of int
        Particle IDs of second halo.
    mass2 : float
        Mass of (particles of) second halo.

    Returns
    -------
    found_match : bool
        True iff halos match.
    """

    frac_parts = 0.5    # Min fraction of shared particles in a match
    frac_mass = 3   # Limit for mass difference between matching halos 

    found_match = False
    shared_parts = np.intersect1d(IDs1,IDs2)
    if (len(shared_parts)/len(IDs1) > frac_parts) and \
            (mass1/mass2 < frac_mass) and \
            (mass1/mass2 > 1/frac_mass):
        found_match = True

    return found_match

def get_subhalo(snap,attr,gn,sgn):
    """ Read snapshot for a halo and return given attribute and index.
    """

    fnum = snap.file_of_halo(gn,sgn)

    # Get index of halo:
    gns = snap.get_subhalos(\
            'GroupNumber',divided=False,fnums=[fnum])[0]
    sgns = snap.get_subhalos(\
            'SubGroupNumber',divided=False,fnums=[fnum])[0]
    idx = np.argwhere(np.logical_and((gns==gn),(sgns==sgn)))

    data = snap.get_subhalos(attr,divided=False,fnums=[fnum])[0]

    return (data[idx],idx)

def get_subhalo_IDs(snap,gn,sgn):

    fnum = snap.file_of_halo(gn,sgn)

    # Get index of halo:
    gns = snap.get_subhalos(\
            'GroupNumber',divided=False,fnums=[fnum])[0]
    sgns = snap.get_subhalos(\
            'SubGroupNumber',divided=False,fnums=[fnum])[0]
    idx = np.argwhere(np.logical_and((gns==gn),(sgns==sgn)))[0,0]

    IDs = snap.get_subhalos_IDs(fnums=[fnum])

    return IDs[idx], idx

