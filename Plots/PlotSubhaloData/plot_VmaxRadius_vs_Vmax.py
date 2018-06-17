import sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from read_subhaloData import read_subhaloData

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

class plot_Rmax_vs_Vmax:

    def __init__(self, dataset='LR'):
        self.dataset = dataset
        maxVelocities = read_subhaloData('Vmax', dataset=self.dataset) / 100000   # cm/s to km/s
        maxRadii = read_subhaloData('VmaxRadius', dataset=self.dataset) * u.cm.to(u.kpc)
        self.stellarMasses = read_subhaloData('Stars/Mass', dataset=self.dataset) * u.g.to(u.Msun)
        self.subGroupNumbers = read_subhaloData('SubGroupNumber', dataset=self.dataset)

        maskSat = np.logical_and.reduce((maxVelocities > 0, maxRadii > 0, self.subGroupNumbers != 0, self.stellarMasses > 0))
        maskIsol = np.logical_and.reduce((maxVelocities > 0, maxRadii > 0, self.subGroupNumbers == 0, self.stellarMasses > 0))

        self.maxVelocitiesSat = maxVelocities[maskSat]
        self.maxRadiiSat = maxRadii[maskSat]
        self.maxVelocitiesIsol = maxVelocities[maskIsol]
        self.maxRadiiIsol = maxRadii[maskIsol]

    def plot(self):
        fig = plt.figure()
        axes = plt.gca()

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.scatter(self.maxVelocitiesSat, self.maxRadiiSat, s=3, c='red', edgecolor='none', label='satellite galaxies')
        axes.scatter(self.maxVelocitiesIsol, self.maxRadiiIsol, s=3, c='blue', edgecolor='none', label='isolated galaxies')

        axes.legend()
        axes.set_xlabel('$v_{max}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$r_{max}[\mathrm{kpc}]$')

        plt.show()
#        plt.savefig('rmax_vs_vmax%s.png'%self.dataset)
#        plt.close()

slice = plot_Rmax_vs_Vmax(dataset='MR')
slice.plot()

