import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from read_subhaloData_MR import read_subhaloData
from read_header_MR import read_header

class plot_subhalo_dist_vs_vmax:

    def __init__(self):
        maxVelocities = read_subhaloData('Vmax')
        stellarMasses = read_subhaloData('Stars/Mass')
        self.maxVelocitiesLum = maxVelocities[np.logical_and(maxVelocities > 0, stellarMasses > 0)]
        self.maxVelocitiesDark = maxVelocities[np.logical_and(maxVelocities > 0, stellarMasses == 0)]

        print(self.maxVelocitiesDark.size)
        #Sort arrays in descending order:
        self.maxVelocitiesLum[::-1].sort()
        self.maxVelocitiesDark[::-1].sort()


    def plot(self):
        fig = plt.figure()
        axes = plt.gca()

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.plot(self.maxVelocitiesLum, np.arange(1, self.maxVelocitiesLum.size + 1), c='black')
        axes.plot(self.maxVelocitiesDark, np.arange(1, self.maxVelocitiesDark.size + 1), c='grey')

        axes.set_xlabel('$v_{max} [\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$N(>v_{max})$')
        axes.set_title('Distribution of luminous subhaloes as a function of $v_{max}$')

        plt.savefig('dist-of-subhaloes-as-function-of-Vmax_MR.png')
        plt.close()

slice = plot_subhalo_dist_vs_vmax() 
slice.plot()

