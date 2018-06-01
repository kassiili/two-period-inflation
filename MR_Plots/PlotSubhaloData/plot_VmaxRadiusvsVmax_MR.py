import numpy as np
import h5py
import matplotlib.pyplot as plt
from read_subhaloData_MR import read_subhaloData
from read_header_MR import read_header

class plot_Rmax_vs_Vmax:

    def __init__(self):
        #self.a, self.h, mass, self.boxsize = read_header() 
        maxVelocities = read_subhaloData('Vmax')
        maxRadii = read_subhaloData('VmaxRadius')*1000 #Mpc -> kpc
        self.subGroupNumbers = read_subhaloData('SubGroupNumber')
        self.stellarMasses = read_subhaloData('Stars/Mass')

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
        plt.savefig('rmax_vs_vmax_MR.png')
        plt.close()

slice = plot_Rmax_vs_Vmax()
slice.plot()

