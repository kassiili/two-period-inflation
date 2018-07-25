import sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
from calc_median import calc_median_trend

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/ReadData/')
from read_header import read_header
from read_subhaloData import read_subhaloData

class plot_SM_vs_Vmax:

    def __init__(self, dataset='LR'):
        self.dataset = dataset
        maxVelocities = read_subhaloData('Vmax', dataset=self.dataset) / 100000   # cm/s to km/s
        stellarMasses = read_subhaloData('Stars/Mass', dataset=self.dataset) * u.g.to(u.Msun)
        self.subGroupNumbers = read_subhaloData('SubGroupNumber', dataset=self.dataset)
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
        axes.scatter(median[0], median[1], s=5)

        median = calc_median_trend(self.maxVelocitiesIsol, self.stellarMassesIsol)
        axes.plot(median[0], median[1], c='blue', linestyle='--')

        axes.legend()
        axes.set_xlabel('$v_{max}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$M_*[\mathrm{M_\odot}]$')
        #axes.set_title('Stellar mass of luminous subhaloes')

        plt.show()
        fig.savefig('Figures/SM_vs_Vmax_%s.png'%self.dataset)
#        plt.close()


plot = plot_SM_vs_Vmax(dataset='MR') 
plot.plot()

