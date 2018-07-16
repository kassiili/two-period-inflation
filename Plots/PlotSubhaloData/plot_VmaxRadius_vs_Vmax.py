import sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from read_subhaloData import read_subhaloData
from calc_median import calc_median_trend

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
        axes.scatter(self.maxVelocitiesSat, self.maxRadiiSat, s=3, c='red', edgecolor='none', label='satelliittigalaksit')
        axes.scatter(self.maxVelocitiesIsol, self.maxRadiiIsol, s=3, c='blue', edgecolor='none', label='eristetyt galaksit')

        median = calc_median_trend(self.maxVelocitiesSat, self.maxRadiiSat)
        axes.plot(median[0], median[1], c='red', linestyle='--')

        median = calc_median_trend(self.maxVelocitiesIsol, self.maxRadiiIsol)
        axes.plot(median[0], median[1], c='blue', linestyle='--')

        plt.xlim(20, 150);
        plt.ylim(1, 50);

        axes.legend(loc=4)
        axes.set_xlabel('$v_{\mathrm{max}}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$r_{\mathrm{max}}[\mathrm{kpc}]$')

#        plt.show()
        plt.savefig('Figures/rmax_vs_vmax_%s.png'%self.dataset)
        plt.close()

slice = plot_Rmax_vs_Vmax(dataset='MR')
slice.plot()

