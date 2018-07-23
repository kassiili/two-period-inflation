import sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
from read_subhaloData import read_subhaloData

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

class plot_subhalo_dist_vs_vmax:

    def __init__(self, dataset='LR'):
        self.dataset = dataset

        gns = read_subhaloData('GroupNumber', dataset=self.dataset)
        sgns = read_subhaloData('SubGroupNumber', dataset=self.dataset)
        COPs = read_subhaloData('CentreOfPotential', dataset=self.dataset) * u.cm.to(u.kpc)
        SM = read_subhaloData('Stars/Mass', dataset=self.dataset)
        mass = read_subhaloData('Mass', dataset=self.dataset) * u.g.to(u.Msun)

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

        axes.legend()
        axes.set_xlabel('$r[\mathrm{kpc}]$')
        axes.set_ylabel('$N(<r)/N_\mathrm{tot}$')
        #axes.set_title('Distribution of luminous subhaloes as a function of $v_{max}$')

        plt.show()
#        plt.savefig('Figures/Dist-of-subhaloes_vs_Vmax_%s.png'%self.dataset)
#        plt.close()

plot = plot_subhalo_dist_vs_vmax(dataset='MR') 
plot.plot()

