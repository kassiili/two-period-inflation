import os, sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class plot_subhalo_dist_by_distance:

    def __init__(self, dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192):

        self.dataset = dataset
        self.nfiles_part = nfiles_part
        self.nfiles_group = nfiles_group
        self.reader = read_data.read_data(dataset=self.dataset, nfiles_part=self.nfiles_part, nfiles_group=self.nfiles_group)

        gns = self.reader.read_subhaloData('GroupNumber')
        sgns = self.reader.read_subhaloData('SubGroupNumber')
        COPs = self.reader.read_subhaloData('CentreOfPotential') * u.cm.to(u.kpc)
        SM = self.reader.read_subhaloData('Stars/Mass')
        mass = self.reader.read_subhaloData('Mass') * u.g.to(u.Msun)

        # Get COP of the central halo:
        centre = COPs[np.logical_and(gns == 1, sgns == 0)]

        # Calculate distances to centre:
        r = np.linalg.norm(COPs - centre, axis=1)

        # Choose subhaloes belonging to the MW analogue and with total mass sufficient for galaxy formation:
        mask = np.logical_and.reduce((gns == 1, sgns != 0, mass > 10**8))
        r = r[mask]
        SM = SM[mask]

        self.rLum = r[SM > 0]
        self.rDark = r[SM == 0]

        # Sort distances in ascending order:
        self.rLum.sort()
        self.rDark.sort()

    def plot(self):
        fig, axes = plt.subplots()

        axes.plot(self.rLum, np.arange(1,self.rLum.size+1)/self.rLum.size, c="black")
        axes.plot(self.rDark, np.arange(1,self.rDark.size+1)/self.rDark.size, c="grey")

        axes.legend(loc=0)
        axes.set_xlabel('$r[\mathrm{kpc}]$')
        axes.set_ylabel('$N(<r)/N_\mathrm{tot}$')
        axes.set_title('Distribution of luminous subhaloes by distance from halo centre\n(%s)'%self.dataset)

        plt.show()
        fig.savefig('../Figures/%s/Radial-dist-of-subhaloes.png'%self.dataset)
        plt.close()

plot = plot_subhalo_dist_by_distance(dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64) 
plot.plot()

