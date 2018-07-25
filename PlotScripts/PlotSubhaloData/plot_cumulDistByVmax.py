import sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/ReadData/')
from read_header import read_header
from read_subhaloData import read_subhaloData

class plot_subhalo_dist_vs_vmax:

    def __init__(self, dataset='LR'):
        self.dataset = dataset

        maxVelocities = read_subhaloData('Vmax', dataset=self.dataset) / 100000 # cm/s to km/s
        stellarMasses = read_subhaloData('Stars/Mass', dataset=self.dataset) * u.g.to(u.Msun)
        subGroupNumbers = read_subhaloData('SubGroupNumber', dataset=self.dataset)
        self.maxVelocitiesSatLum = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, subGroupNumbers != 0))]
        self.maxVelocitiesSatDark = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses == 0, subGroupNumbers != 0))]
        self.maxVelocitiesIsolLum = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, subGroupNumbers == 0))]
        self.maxVelocitiesIsolDark = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses == 0, subGroupNumbers == 0))]

        #Sort arrays in descending order:
        self.maxVelocitiesSatLum[::-1].sort()
        self.maxVelocitiesSatDark[::-1].sort()
        self.maxVelocitiesIsolLum[::-1].sort()
        self.maxVelocitiesIsolDark[::-1].sort()

    def plot(self):
        fig = plt.figure()
        axes = plt.gca()

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.plot(self.maxVelocitiesSatLum, np.arange(1, self.maxVelocitiesSatLum.size + 1), c='lightblue', label='Kirkkaat satelliittigalaksit')
        axes.plot(self.maxVelocitiesSatDark, np.arange(1, self.maxVelocitiesSatDark.size + 1), c='blue', label='Pimeät satelliittigalaksit')
        axes.plot(self.maxVelocitiesIsolLum, np.arange(1, self.maxVelocitiesIsolLum.size + 1), c='pink', label='Kirkkaat eristetyt galaksit')
        axes.plot(self.maxVelocitiesIsolDark, np.arange(1, self.maxVelocitiesIsolDark.size + 1), c='red', label='Pimeät eristetyt galaksit')

        axes.legend()
        axes.set_xlabel('$v_{\mathrm{max}} [\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$N(>v_{\mathrm{max}})$')
        #axes.set_title('Distribution of luminous subhaloes as a function of $v_{max}$')

#        plt.show()
        plt.savefig('Figures/Dist-of-subhaloes_vs_Vmax_%s.png'%self.dataset)
        plt.close()

plot = plot_subhalo_dist_vs_vmax(dataset='MR') 
plot.plot()

