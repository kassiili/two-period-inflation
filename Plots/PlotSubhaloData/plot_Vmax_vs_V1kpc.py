import sys
import numpy as np
import h5py
import time
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from read_subhaloData import read_subhaloData

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/PlotPartPos/')
from read_dataset import read_dataset
from read_dataset_dm_mass import read_dataset_dm_mass

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

class plot_Vmax_vs_V1kpc:

    def calcVelocitiesAt1kpc(self):
        """ For each subhalo, calculate the circular velocity at 1kpc. """

        massWithin1kpc = np.zeros((self.subhaloData['groupNumber'].size))

        for idx, (gn, sgn, cop) in enumerate(zip(self.subhaloData['groupNumber'], self.subhaloData['subGroupNumber'], self.subhaloData['COP'])):

            # Get coordinates and corresponding masses of the particles in the halo:
            mask = np.logical_and(self.combined['groupNumber'] == gn, self.combined['subGroupNumber'] == sgn)
            coords = self.combined['coords'][mask]
            mass = self.combined['mass'][mask]

            # Periodic wrap coordinates around centre.
#            boxsize = self.boxsize/self.h
#            data['coords'] = np.mod(data['coords']-self.centre+0.5*boxsize,boxsize)+self.centre-0.5*boxsize

            # Calculate distances from COP:
            r = np.linalg.norm(coords - cop, axis=1)

            massWithin1kpc[idx] = mass[np.logical_and(r > 0, r < 1)].sum()

        myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value

        return np.sqrt(massWithin1kpc * myG)

    def __init__(self, dataset='LR'):

        self.dataset = dataset
        self.a, self.h, self.massTable, self.boxsize = read_header() 

        # Read particle data of all particle types:
        gas = self.read_particle_data(0)
        dm = self.read_particle_data(1)
        stars = self.read_particle_data(4)
        BHs = self.read_particle_data(5)

        # Combine particle data into single array:
        self.combined = {}
        self.combined['coords'] = np.vstack((gas['coords'], dm['coords'], stars['coords'], BHs['coords']))
        self.combined['mass'] = np.concatenate((gas['mass'], dm['mass'], stars['mass'], BHs['mass']))
        self.combined['groupNumber'] = np.concatenate((gas['groupNumber'], dm['groupNumber'], stars['groupNumber'], BHs['groupNumber']))
        self.combined['subGroupNumber'] = np.concatenate((gas['subGroupNumber'], dm['subGroupNumber'], stars['subGroupNumber'], BHs['subGroupNumber']))

        self.subhaloData = self.read_subhalo_data()

        start = time.clock()
        self.velocitiesAt1kpc = self.calcVelocitiesAt1kpc()
        print(time.clock() - start)
        
        maskSat = np.logical_and(self.subhaloData['Vmax'] > 0, self.subhaloData['subGroupNumber'] != 0)
        maskIsol = np.logical_and(self.subhaloData['Vmax'] > 0, self.subhaloData['subGroupNumber'] == 0)
        self.VmaxSat = self.subhaloData['Vmax'][maskSat]
        self.V1kpcSat = self.velocitiesAt1kpc[maskSat]
        self.VmaxIsol = self.subhaloData['Vmax'][maskIsol]
        self.V1kpcIsol = self.velocitiesAt1kpc[maskIsol]

    def read_particle_data(self, part_type):
        """ Read group numbers, subgroup numbers, particle masses and coordinates of all the particles of a certain type. """

        data = {}

        data['groupNumber']  = read_dataset(part_type, 'GroupNumber')
        data['subGroupNumber'] = read_dataset(part_type, 'SubGroupNumber')

        if part_type == 1:
            data['mass'] = read_dataset_dm_mass() * u.g.to(u.Msun)
        else:
            data['mass'] = read_dataset(part_type, 'Masses') * u.g.to(u.Msun)
        data['coords'] = read_dataset(part_type, 'Coordinates') * u.cm.to(u.kpc)

        return data

    def read_subhalo_data(self):
        """ Read group numbers, subgroup numbers, max circular velocities and centre of potentials of all the subhaloes. """

        data = {}

        data['groupNumber'] = read_subhaloData('GroupNumber')
        data['subGroupNumber'] = read_subhaloData('SubGroupNumber')

        data['Vmax'] = read_subhaloData('Vmax')/100000  # cm/s to km/s
        data['COP'] = read_subhaloData('CentreOfPotential') * u.cm.to(u.kpc)

        return data

    def plot(self):
        fig = plt.figure()
        axes = plt.gca()

        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.scatter(self.V1kpcSat[self.V1kpcSat > 10], self.VmaxSat[self.V1kpcSat > 10], s=3, c='red', edgecolor='none', label='satellite galaxies')
        axes.scatter(self.V1kpcIsol[self.V1kpcIsol > 10], self.VmaxIsol[self.V1kpcIsol > 10], s=3, c='blue', edgecolor='none', label='isolated galaxies')

#        start = time.clock()
#        median = self.calc_median_trend2(self.maxVelocitiesSat, self.stellarMassesSat)
#        axes.plot(median[0], median[1], c='red', linestyle='--')

        mask = self.velocitiesAt1kpc > self.subhaloData['Vmax']
        oddOnes = {}
        oddOnes['SGN'] = self.subhaloData['subGroupNumber'][mask]
        oddOnes['GN'] = self.subhaloData['groupNumber'][mask]

        for gn, sgn in zip(oddOnes['GN'], oddOnes['SGN']):
            mask2 = np.logical_and(self.subhaloData['groupNumber'] == gn, self.subhaloData['subGroupNumber'] == sgn)
            axes.scatter(self.velocitiesAt1kpc[mask2], self.subhaloData['Vmax'][mask2], s=5, c='green', edgecolor='none')
            print(gn, sgn)

        x = np.arange(10,100)
        y = np.arange(10,100)
        axes.plot(x, y, c='black')

        axes.legend()
        axes.set_xlabel('$v_{1kpc}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$v_{max}[\mathrm{km s^{-1}}]$')

        plt.show()
#        plt.savefig('Vmax_vs_V1kpc_%s.png'%self.dataset)
#        plt.close()


plot = plot_Vmax_vs_V1kpc() 
plot.plot()
