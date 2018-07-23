import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
from read_subhaloData import read_subhaloData

def calc_satellites(dataset='LR'):

    sgns = read_subhaloData('SubGroupNumber', dataset=dataset)
    gns = read_subhaloData('GroupNumber', dataset=dataset)
    COPs = read_subhaloData('CentreOfPotential', dataset=dataset) * u.cm.to(u.kpc)
    masses = read_subhaloData('Mass', dataset=dataset) * u.g.to(u.Msun)
    SM = read_subhaloData('Stars/Mass', dataset=dataset) * u.g.to(u.Msun)

    # Calculate barycentre and mass of the LG-analogue:
    MW_mask = np.logical_and(gns == 1, sgns == 0)
    M31_mask = np.logical_and(gns == 2, sgns == 0)
    mass = masses[MW_mask] + masses[M31_mask]
    centre = (masses[MW_mask] * COPs[MW_mask] + masses[M31_mask] * COPs[M31_mask]) / mass

    # Calculate distances to the barycentre:
    r = np.linalg.norm(COPs - centre, axis=1)

    # Return number of satellites within 300 kpc from the barycentre:
    return np.sum(np.logical_and.reduce((sgns != 0, r < 300, SM > 10**5))), mass

N, mass = calc_satellites(dataset='MR')

plt.figure()
plt.scatter(mass / 10**12, N)
plt.xlabel('$M_{200}[\mathrm{10^{12} M_\odot}]$')
plt.ylabel('$N(r<300\mathrm{ kpc})$')

plt.show()
plt.close()
