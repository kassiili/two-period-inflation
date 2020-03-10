import numpy as np
import h5py
from collections import deque

import astropy.units as u
from astropy.constants import G

from snapshot_obj import Snapshot

def calculate_attr(dataset,attr):
    """ An interface to the functions of this module. Uses the correct
    function to construct the dataset corresponding to attr in dataset.
    """

    if attr == 'V1kpc':
        return calculate_V1kpc(dataset)

    return None

def calculate_V1kpc_slow(dataset):
    """ For each subhalo, calculate the circular velocity at 1kpc. """

    # Get particle data:
    part = {}
    part['gns'] = dataset.get_particles('GroupNumber')
    part['sgns'] = dataset.get_particles('SubGroupNumber')
    part['coords'] = dataset.get_particles('Coordinates')\
            * u.cm.to(u.kpc)
    part['mass'] = dataset.get_particle_masses() * u.g.to(u.Msun)

    # Get subhalodata:
    halo = {}
    halo['gns'] = dataset.get_subhalos('GroupNumber',divided=False)[0]
    halo['sgns'] = \
            dataset.get_subhalos('SubGroupNumber',divided=False)[0]
    halo['COPs'] = dataset.get_subhalos(\
            'CentreOfPotential',divided=False)[0] * u.cm.to(u.kpc)

    massWithin1kpc = np.zeros((halo['gns'].size))

    # Loop through subhalos:
    for idx, (gn, sgn, cop) in \
            enumerate(zip(halo['gns'],halo['sgns'],halo['COPs'])):

        # Get coordinates and masses of the particles in the halo:
        halo_mask = np.logical_and(part['gns'] == gn, \
                part['sgns'] == sgn)
        coords = part['coords'][halo_mask]
        mass = part['mass'][halo_mask]

        # Calculate distances to COP:
        r = np.linalg.norm(coords - cop, axis=1)

        # Get coordinates within 1kpc from COP:
        r1kpc_mask = np.logical_and(r > 0, r < 1)

        massWithin1kpc[idx] = mass[r1kpc_mask].sum()

    myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
    v1kpc = np.sqrt(massWithin1kpc * myG)

    return v1kpc

def calculate_V1kpc(dataset):
    """ For each subhalo, calculate the circular velocity at 1kpc. 
    Assume that there are no jumps in the SubGroupNumber values in any
    of the groups."""

    # Get particle data:
    coords = dataset.get_particles('Coordinates') \
            * u.cm.to(u.kpc)
    mass = dataset.get_particle_masses() * u.g.to(u.Msun)

    # Get halo data:
    COPs = dataset.get_subhalos('CentreOfPotential',\
            divided=False)[0] * u.cm.to(u.kpc)
    part_idx = get_subhalo_part_idx(dataset)

    massWithin1kpc = np.zeros((COPs[:,0].size))

    for idx, (cop,idx_list) in enumerate(zip(COPs,part_idx)):

        # Get coords and mass of the particles in the corresponding halo:
        halo_coords = coords[idx_list]
        halo_mass = mass[idx_list]

        # Calculate distances to COP:
        r = np.linalg.norm(halo_coords - cop, axis=1)

        # Get coordinates within 1kpc from COP:
        r1kpc_mask = np.logical_and(r > 0, r < 1)

        massWithin1kpc[idx] = halo_mass[r1kpc_mask].sum()

    myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
    v1kpc = np.sqrt(massWithin1kpc * myG)

    return v1kpc
    
def get_subhalo_part_idx(dataset):
    """ Finds indices of the particles in each halo. """

    # Get subhalos:
    halo_gns = dataset.get_subhalos('GroupNumber',\
            divided=False)[0].astype(int)
    halo_sgns = dataset.get_subhalos('SubGroupNumber',\
            divided=False)[0].astype(int)

    # Get particles:
    part_gns = dataset.get_particles('GroupNumber')
    part_sgns = dataset.get_particles('SubGroupNumber')

    # Get halo indices:
    sorting = np.lexsort((halo_sgns,halo_gns))
    print(halo_gns.size)

    # Invert sorting:
    inv_sorting = [0] * len(sorting)
    for idx, val in enumerate(sorting):
        inv_sorting[val] = idx

    # Loop through particles and save indices to lists. Halos in the
    # list behind the part_idx key are arranged in ascending order 
    # with gn and sgn, i.e. in the order lexsort would arrange them:
    gn_count = np.bincount(halo_gns)
    print(halo_gns.size + sum(gn_count==0))
    print(gn_count.sum() + sum(gn_count==0))
    part_idx = [[] for i in range(halo_gns.size + sum(gn_count==0))]
    for idx, (gn,sgn) in enumerate(zip(part_gns,part_sgns)):
        # Exclude unbounded particles (for unbounded: sgn = max_int):
        if sgn < 10**6:
            i = gn_count[:gn].sum()+sgn
            if i >= len(part_idx):
                print(i)
                print(gn,sgn)
            part_idx[i].append(idx)

    print(len(part_idx))
    # Strip from empty lists:
    part_idx = [l for l in part_idx if not (not l)]
    print(len(part_idx))

    # Convert to ndarray and sort in order corresponding to the halo
    # datasets:
    part_idx = np.array(part_idx)[inv_sorting]

    return part_idx

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
    """_


