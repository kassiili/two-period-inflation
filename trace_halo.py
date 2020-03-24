import numpy as np
import h5py
import math
from collections import deque

from snapshot_obj import Snapshot

def trace_halo(snap_init,gn,sgn,direction='forward',stop=101):
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
    direction : str, optional
        Direction in which halo is traced.
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

    # Initialize tracer:
    tracer = {snap_init.snapID : (z_init,gn,sgn)}
    snap = snap_init

    if direction == 'forward':
        condition = lambda ID : ID < 127
        add = 1
    else:
        condition = lambda ID : ID > stop
        add = -1

    while condition(snap.snapID):
        snap_next = Snapshot(snap.simID,snap.snapID+add)
        gn, sgn = match_snapshots(snap_next, snap, gn, sgn)

        # No matching halo found:
        if gn == -1: break

        with h5py.File(snap_next.grp_file,'r') as grpf:
            z_next = grpf['link0/Header'].attrs.get('Redshift')

        # Add match to tracer:
        tracer[snap_next.snapID] = (z_next,gn,sgn)

        snap = snap_next

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
    match : tuple
        (gn,sgn) of matching halo. match==(-1,-1) if no match is found.
    """

    # Get particle IDs and halo mass of reference:
    IDs_ref,_ = get_subhalo_IDs(snap_ref,gn,sgn)
    mass_ref,_ = get_subhalo(snap_ref,'Mass',gn,sgn)

    # Set maximum number of iterations:
    term = 100

    # Read subhalos with group numbers and subgroup numbers near gn and
    # sgn:
    fnums = neighborhood(snap,gn,sgn,term/2)
    GNs = snap.get_subhalos('GroupNumber',fnums=fnums)
    SGNs = snap.get_subhalos('SubGroupNumber',fnums=fnums)
    IDs_in_file = snap.get_subhalos_IDs(fnums=fnums)
    mass_in_file = snap.get_subhalos('Mass',fnums=fnums)

    # Get index of halo with same sgn and gn as ref:
    idx0 = np.argwhere(np.logical_and((GNs==gn),(SGNs==sgn)))[0,0]
    print('find match for:',GNs[idx0],SGNs[idx0],' in ',snap.snapID)

    # Initial value of match is returned if no match is found:
    match = (-1,-1)

    idx = idx0
    for step in range(1,term):
        print('idx=',idx)
        IDs = IDs_in_file[idx]; mass = mass_in_file[idx]
        found_match = match_subhalos(IDs_ref,mass_ref,IDs,mass)

        if found_match:
            match = (GNs[idx],SGNs[idx])
            break

        # Iterate outwards from idx0, alternating between lower and higher
        # index:
        idx = idx0 + int(math.copysign(\
                math.floor((step+1)/2), (step % 2) - 0.5))

    return match

def neighborhood(snap,gn,sgn,min_halos):
    """ Gets file numbers of files that contain a minimum amount of
    halos above and below a certain halo. """

    fnum = snap.file_of_halo(gn,sgn)
    GNs = snap.get_subhalos('GroupNumber',fnums=[fnum])
    SGNs = snap.get_subhalos('SubGroupNumber',fnums=[fnum])
    idx = np.argwhere(np.logical_and((GNs==gn),(SGNs==sgn)))[0,0]

    fnums = [fnum]
    halos_below = idx
    for n in range(fnum-1,-1,-1):
        if halos_below < min_halos:
            fnums.append(n)
            halos_below += snap.get_subhalos('GroupNumber',fnums=[n]).size
        else:
            break

    with h5py.File(snap.grp_file,'r') as grpf:
        n_files = grpf['link1/FOF'].attrs.get('NTask')

    halos_above = GNs.size-1-idx
    for n in range(fnum+1,n_files):
        if halos_above < min_halos:
            fnums.append(n)
            halos_above += snap.get_subhalos('GroupNumber',fnums=[n]).size
        else:
            break

    return fnums

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

    frac_parts = 0.3    # Min fraction of shared particles in a match
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
    gns = snap.get_subhalos('GroupNumber',fnums=[fnum])
    sgns = snap.get_subhalos('SubGroupNumber',fnums=[fnum])
    idx = np.argwhere(np.logical_and((gns==gn),(sgns==sgn)))[0,0]

    data = snap.get_subhalos(attr,fnums=[fnum])

    return (data[idx],idx)

def get_subhalo_IDs(snap,gn,sgn):
    """ Read snapshot for a halo and return IDs of its particles and index.
    """

    fnum = snap.file_of_halo(gn,sgn)

    # Get index of halo:
    gns = snap.get_subhalos('GroupNumber',fnums=[fnum])
    sgns = snap.get_subhalos('SubGroupNumber',fnums=[fnum])
    idx = np.argwhere(np.logical_and((gns==gn),(sgns==sgn)))[0,0]

    IDs = snap.get_subhalos_IDs(fnums=[fnum])

    return IDs[idx], idx

