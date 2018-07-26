import os, sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
from calc_median import calc_median_trend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class plot_SM_vs_Vmax:

    def __init__(self, dataset='V1_LR_fix_127_z000p000', nfiles_part=16, nfiles_group=96):

        self.dataset = dataset
        self.nfiles_part = nfiles_part
        self.nfiles_group = nfiles_group
        self.reader = read_data.read_data(dataset=self.dataset, nfiles_part=self.nfiles_part, nfiles_group=self.nfiles_group)

        maxVelocities = self.reader.read_subhaloData('Vmax') / 100000   # cm/s to km/s
        stellarMasses = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
        self.subGroupNumbers = self.reader.read_subhaloData('SubGroupNumber')
        maskSat = np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, self.subGroupNumbers != 0))
        maskIsol = np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, self.subGroupNumbers == 0))

        self.maxVelocitiesSat = maxVelocities[maskSat]
        self.stellarMassesSat = stellarMasses[maskSat]
        self.maxVelocitiesIsol = maxVelocities[maskIsol]
        self.stellarMassesIsol = stellarMasses[maskIsol]

    def plot(self):
        fig, axes = plt.subplots()

        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.scatter(self.maxVelocitiesSat, self.stellarMassesSat, s=3, c='red', edgecolor='none', label='satellite galaxies')
        axes.scatter(self.maxVelocitiesIsol, self.stellarMassesIsol, s=3, c='blue', edgecolor='none', label='isolated galaxies')

        median = calc_median_trend(self.maxVelocitiesSat, self.stellarMassesSat, points_per_bar=7)
        axes.plot(median[0], median[1], c='red', linestyle='--')
        #axes.scatter(median[0], median[1], s=5)

        median = calc_median_trend(self.maxVelocitiesIsol, self.stellarMassesIsol)
        axes.plot(median[0], median[1], c='blue', linestyle='--')

        axes.set_xlim(10, 100)
        axes.set_ylim(10**6, 10**10)
        axes.legend(loc=0)
        axes.set_xlabel('$v_{max}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$M_*[\mathrm{M_\odot}]$')
        axes.set_title('Stellar mass of luminous subhaloes')

        #plt.show()
        fig.savefig('../Figures/%s/SM_vs_Vmax.png'%self.dataset)
        plt.close()


plot = plot_SM_vs_Vmax(dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192)
plot.plot()

