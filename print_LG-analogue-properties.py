import os,sys
import numpy as np
import astropy.units as u

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ReadData"))
from read_data import read_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PlotSubhaloData"))
from dataset import dataset

dataset = dataset('V1_MR_fix_127_z000p000', 'standard', 16, 192)
reader = read_data(dataset=dataset.dir, nfiles_part=dataset.nfiles_part, nfiles_group=dataset.nfiles_group)
a, h, mass, boxsize = reader.read_header()
h = h * u.km.to(u.Mpc) # (km/s)/Mpc -> 1/s

# Read halo identifiers:
GNs = reader.read_subhaloData('GroupNumber')
SGNs = reader.read_subhaloData('SubGroupNumber')

# Get MW and Andromeda masses:
mask1 = np.logical_and(SGNs == 0, GNs == 1)
mask2 = np.logical_and(SGNs == 0, GNs == 2)
mass1 = reader.read_subhaloData('Mass')[mask1] * u.g.to(u.Msun)
mass2 = reader.read_subhaloData('Mass')[mask2] * u.g.to(u.Msun)
if (mass1 > mass2):
    mask_M31 = mask1; mask_MW = mask2
    mass_M31 = mass1; mass_MW = mass2
else:
    mask_M31 = mask2; mask_MW = mask1
    mass_M31 = mass2; mass_MW = mass1

print('M31, mass:', mass_M31)    
print('MW, mass:', mass_MW)    

# Print distance:
COPs = reader.read_subhaloData('CentreOfMass') * u.cm.to(u.kpc)
r = COPs[mask1] - COPs[mask2]
r_norm = np.linalg.norm(r)
print('Distance: ', r_norm)

# Print radial velocity:
v_MW = reader.read_subhaloData('Velocity')[mask_MW] * u.cm.to(u.km)
v_M31 = reader.read_subhaloData('Velocity')[mask_M31] * u.cm.to(u.km)
r = r * u.kpc.to(u.km); r_norm = r_norm * u.kpc.to(u.km)
r_unit = r / r_norm
v_rad_pec = np.inner(v_MW-v_M31,r_unit)
v_rad = v_rad_pec + a*h*r_norm
print('Radial velocity: ', v_rad)
