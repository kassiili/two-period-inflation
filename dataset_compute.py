import numpy as np
import h5py

import astropy.units as u
from astropy.constants import G

def split_satellites(snap, attr, fnums=[]):
    """ Reads an attribute from snapshot and divides into satellites and
    isolated galaxies.
    
    Parameters
    ----------
    attr : str
        attribute to be retrieved
    fnums : list of ints, optional
        Specifies files, which are to be read

    Returns
    -------
    data : tuple of HDF5 datasets
        Satellite data in the first entry and isolated galaxies data in
        the second.
    """

    SGNs = snap.get_subhalos('SubGroupNumber', fnums=fnums)
    data = snap.get_subhalos(attr, fnums=fnums)

    # Divide into satellites and isolated galaxies:
    dataSat = data[SGNs != 0]
    dataIsol = data[SGNs == 0]

    return (dataSat,dataIsol)

def generate_dataset(snapshot,dataset):
    """ An interface to the functions of this module. Uses the correct
    function to construct the dataset.
    """

    if dataset == 'V1kpc':
        return compute_V1kpc(snapshot)

    return None

def compute_V1kpc(snapshot):
    """ For each subhalo, calculate the circular velocity at 1kpc. """

    # Get particle data:
    part = {}
    part['gns'] = snapshot.get_particles('GroupNumber')
    part['sgns'] = snapshot.get_particles('SubGroupNumber')
    part['coords'] = snapshot.get_particles('Coordinates')\
            * u.cm.to(u.kpc)
    part['mass'] = snapshot.get_particle_masses() * u.g.to(u.Msun)

    # Get subhalodata:
    halo = {}
    halo['gns'] = snapshot.get_subhalos('GroupNumber')
    halo['sgns'] = snapshot.get_subhalos('SubGroupNumber')
    halo['COPs'] = snapshot.get_subhalos('CentreOfPotential') \
            * u.cm.to(u.kpc)

    # Get box size:
    with h5py.File(snapshot.part_file,'r') as partf:
        h = partf['link1/Header'].attrs.get('HubbleParam')
        boxs = partf['link1/Header'].attrs.get('BoxSize') * 1000/h 
                                                            # Mpc/h -> kpc

    massWithin1kpc = np.zeros((halo['gns'].size))

    # Loop through subhalos:
    for idx, (gn, sgn, cop) in \
            enumerate(zip(halo['gns'],halo['sgns'],halo['COPs'])):

        # Get coordinates and masses of the particles in the halo:
        halo_mask = np.logical_and(part['gns'] == gn, \
                part['sgns'] == sgn)
        coords = part['coords'][halo_mask]
        mass = part['mass'][halo_mask]

        # Calculate distances to cop:
        d = np.mod(coords-cop+0.5*boxs, boxs) - 0.5*boxs
        r = np.linalg.norm(d, axis=1)

        # Get coordinates within 1kpc from COP:
        r1kpc_mask = np.logical_and(r > 0, r < 1)

        massWithin1kpc[idx] = mass[r1kpc_mask].sum()

    myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
    v1kpc = np.sqrt(massWithin1kpc * myG)

    return v1kpc

def compute_rotation_curve(snapshot, gn, sgn, part_type=[0,1,4,5], jump=10):
    """ Compute circular velocity by radius for a given halo. 
    
    Parameters
    ----------
    jump : int, optional
        Reduce noise by only computing circular velocity at the location
        of every nth particle, where n = jump
    Returns
    -------
    (r,v_circ) : tuple
        Circular velocities, v_circ, at distances r to halo centre in
        cgs units.
    """

    # Get centre of potential:
    SGNs = snapshot.get_subhalos("SubGroupNumber")
    GNs = snapshot.get_subhalos("GroupNumber")
    COPs = snapshot.get_subhalos("CentreOfPotential")

    halo_mask = np.logical_and(SGNs == sgn,GNs == gn)
    cop = COPs[halo_mask]

    # Get coordinates and masses of the halo:
    SGNs = snapshot.get_particles("SubGroupNumber",part_type=part_type)
    GNs = snapshot.get_particles("GroupNumber",part_type=part_type)
    coords = snapshot.get_particles("Coordinates",part_type=part_type)
    mass = snapshot.get_particle_masses(part_type=part_type)
    
    halo_mask = np.logical_and(SGNs == sgn,GNs == gn)
    coords = periodic_wrap(snapshot,cop,coords[halo_mask])
    mass = mass[halo_mask]
    
    # Calculate distance to centre and cumulative mass:
    r = np.linalg.norm(coords - cop, axis=1)
    sorting = np.argsort(r)
    r = r[sorting]
    cmass = np.cumsum(mass[sorting])
   
    # Clean up:
    mask = r>0; r=r[mask]; cumass=cmass[mask]
    r = r[jump::jump]
    cumass = cumass[jump::jump]
    
    # Compute velocity.
    myG = G.to(u.cm**3 * u.g**-1 * u.s**-2).value
    v_circ = np.sqrt((myG * cumass) / r)

    return r,v_circ

def periodic_wrap(snapshot,cop,coords):
    """ Account for the periodic boundary conditions by moving particles 
    to the periodic location, which is closest to the cop of their host
    halo. """

    # Periodic wrap coordinates around centre.
    with h5py.File(snapshot.part_file,'r') as partf:
        h = partf['link1/Header'].attrs.get('HubbleParam')
        boxs = partf['link1/Header'].attrs.get('BoxSize')  
        boxs = snapshot.convert_to_cgs_group(np.array([boxs]),\
                'CentreOfPotential') / h
    wrapped = np.mod(coords-cop+0.5*boxs, boxs) + cop-0.5*boxs

    return wrapped

def calculate_V1kpc_inProgress(snapshot):
    """ For each subhalo, calculate the circular velocity at 1kpc. 
    Assume that there are no jumps in the SubGroupNumber values in any
    of the groups."""

    # Get particle data:
    coords = snapshot.get_particles('Coordinates') \
            * u.cm.to(u.kpc)
    mass = snapshot.get_particle_masses() * u.g.to(u.Msun)

    # Get halo data:
    COPs = snapshot.get_subhalos('CentreOfPotential',\
            divided=False)[0] * u.cm.to(u.kpc)
    part_idx = get_subhalo_part_idx(snapshot)

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
    
def get_subhalo_part_idx(snapshot):
    """ Finds indices of the particles in each halo. """

    # Get subhalos:
    halo_gns = snapshot.get_subhalos('GroupNumber',\
            divided=False)[0].astype(int)
    halo_sgns = snapshot.get_subhalos('SubGroupNumber',\
            divided=False)[0].astype(int)

    # Get particles:
    part_gns = snapshot.get_particles('GroupNumber')
    part_sgns = snapshot.get_particles('SubGroupNumber')

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

