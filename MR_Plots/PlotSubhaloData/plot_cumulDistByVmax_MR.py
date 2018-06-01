import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import matplotlib
from read_subhaloData_MR import read_subhaloData
from read_header_MR import read_header

class plot_subhalo_dist_vs_vmax:

    def __init__(self):
        maxVelocities = read_subhaloData('Vmax')
        stellarMasses = read_subhaloData('Stars/Mass')
        subGroupNumbers = read_subhaloData('SubGroupNumber')
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
        axes.plot(self.maxVelocitiesSatLum, np.arange(1, self.maxVelocitiesSatLum.size + 1), c='lightblue', label='Luminous sat. galaxies')
        axes.plot(self.maxVelocitiesSatDark, np.arange(1, self.maxVelocitiesSatDark.size + 1), c='blue', label='Dark sat. galaxies')
        axes.plot(self.maxVelocitiesIsolLum, np.arange(1, self.maxVelocitiesIsolLum.size + 1), c='pink', label='Luminous isol. galaxies')
        axes.plot(self.maxVelocitiesIsolDark, np.arange(1, self.maxVelocitiesIsolDark.size + 1), c='red', label='Dark isol. galaxies')

        axes.legend()
        axes.set_xlabel('$v_{max} [\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$N(>v_{max})$')
        axes.set_title('Distribution of luminous subhaloes as a function of $v_{max}$')

        plt.show()
        plt.savefig('dist-of-subhaloes-as-function-of-Vmax_MR.png')
        plt.close()

slice = plot_subhalo_dist_vs_vmax() 
slice.plot()

