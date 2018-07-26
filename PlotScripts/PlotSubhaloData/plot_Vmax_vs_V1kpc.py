import os, sys
import numpy as np
import h5py
import time
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from calc_median import calc_median_trend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class plot_Vmax_vs_V1kpc:

    def __init__(self, dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192):

        self.dataset = dataset
        self.nfiles_part = nfiles_part
        self.nfiles_group = nfiles_group
        self.reader = read_data.read_data(dataset=self.dataset, nfiles_part=self.nfiles_part, nfiles_group=self.nfiles_group)

        self.a, self.h, self.massTable, self.boxsize = self.reader.read_header() 

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
        self.subhaloData['V1kpc'] = self.calcVelocitiesAt1kpc()

    def calcVelocitiesAt1kpc(self):
        """ For each subhalo, calculate the circular velocity at 1kpc. """

        massWithin1kpc = np.zeros((self.subhaloData['groupNumber'].size))

        for idx, (gn, sgn, cop) in enumerate(zip(self.subhaloData['groupNumber'], self.subhaloData['subGroupNumber'], self.subhaloData['COP'])):

            # Get coordinates and corresponding masses of the particles in the halo:
            halo_mask = np.logical_and(self.combined['groupNumber'] == gn, self.combined['subGroupNumber'] == sgn)
            coords = self.combined['coords'][halo_mask]
            mass = self.combined['mass'][halo_mask]

            # Periodic wrap coordinates around centre.
#            boxsize = self.boxsize/self.h
#            data['coords'] = np.mod(data['coords']-self.centre+0.5*boxsize,boxsize)+self.centre-0.5*boxsize

            # Calculate distances to COP:
            r = np.linalg.norm(coords - cop, axis=1)

            # Get coordinates within 1kpc from COP:
            r1kpc_mask = np.logical_and(r > 0, r < 1)

            # Set V1kpc to -1 for haloes with less than 10 particles within 1kpc:
            if (r1kpc_mask.sum() < 10):
                massWithin1kpc[idx] = -1
            else:
                massWithin1kpc[idx] = mass[r1kpc_mask].sum()

        # Exclude haloes with less than 10 particles within 1kpc:
        noiseReduction_mask = massWithin1kpc > 0

        for key in self.subhaloData.keys():
            self.subhaloData[key] = self.subhaloData[key][noiseReduction_mask]

        myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value

        return np.sqrt(massWithin1kpc[noiseReduction_mask] * myG)
        
    def read_particle_data(self, part_type):
        """ Read group numbers, subgroup numbers, particle masses and coordinates of all the particles of a certain type. """

        data = {}

        data['groupNumber']  = self.reader.read_dataset(part_type, 'GroupNumber')
        data['subGroupNumber'] = self.reader.read_dataset(part_type, 'SubGroupNumber')

        if part_type == 1:
            data['mass'] = self.reader.read_dataset_dm_mass() * u.g.to(u.Msun)
        else:
            data['mass'] = self.reader.read_dataset(part_type, 'Masses') * u.g.to(u.Msun)
        data['coords'] = self.reader.read_dataset(part_type, 'Coordinates') * u.cm.to(u.kpc)

        return data

    def read_subhalo_data(self):
        """ Read group numbers, subgroup numbers, max circular velocities and centre of potentials of all the subhaloes. """

        data = {}

        data['Vmax'] = self.reader.read_subhaloData('Vmax')/100000  # cm/s to km/s

        # Exclude unphysical haloes with Vmax non-positive:
        mask = data['Vmax'] > 0

        data['Vmax'] = data['Vmax'][mask]
        data['COP'] = self.reader.read_subhaloData('CentreOfPotential')[mask] * u.cm.to(u.kpc)
        data['groupNumber'] = self.reader.read_subhaloData('GroupNumber')[mask]
        data['subGroupNumber'] = self.reader.read_subhaloData('SubGroupNumber')[mask]

        return data

    def plot(self):

        # Plot satellites and isolated galaxies separately:
        maskSat = self.subhaloData['subGroupNumber'] != 0
        maskIsol = self.subhaloData['subGroupNumber'] == 0

        VmaxSat = self.subhaloData['Vmax'][maskSat]
        V1kpcSat = self.subhaloData['V1kpc'][maskSat]
        VmaxIsol = self.subhaloData['Vmax'][maskIsol]
        V1kpcIsol = self.subhaloData['V1kpc'][maskIsol]

        fig, axes = plt.subplots()

        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.scatter(V1kpcSat, VmaxSat, s=3, c='red', edgecolor='none', label='satellite galaxies')
        axes.scatter(V1kpcIsol, VmaxIsol, s=3, c='blue', edgecolor='none', label='isolated galaxies')

        median = calc_median_trend(V1kpcSat, VmaxSat)
        axes.plot(median[0], median[1], c='red', linestyle='--')

        median = calc_median_trend(V1kpcIsol, VmaxIsol)
        axes.plot(median[0], median[1], c='blue', linestyle='--')

#        mask = self.subhaloData['V1kpc'] > self.subhaloData['Vmax']
#        oddOnes = {}
#        oddOnes['SGN'] = self.subhaloData['subGroupNumber'][mask]
#        oddOnes['GN'] = self.subhaloData['groupNumber'][mask]
#
#        for gn, sgn in zip(oddOnes['GN'], oddOnes['SGN']):
#            mask2 = np.logical_and(self.subhaloData['groupNumber'] == gn, self.subhaloData['subGroupNumber'] == sgn)
#            axes.scatter(self.subhaloData['V1kpc'][mask2], self.subhaloData['Vmax'][mask2], s=5, c='green', edgecolor='none')
#            print(gn, sgn)

        # Plot identity function. No physically meaningful data point should lay below this line.
        #x = np.arange(10,100)
        #y = np.arange(10,100)
        #axes.plot(x, y, c='black')

        plt.xlim(10, 110)
        plt.ylim(10, 110)

        axes.legend(loc=0)
        axes.set_xlabel('$v_{\mathrm{1kpc}}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$v_{\mathrm{max}}[\mathrm{km s^{-1}}]$')

        plt.show()
        fig.savefig('../Figures/%s/Vmax_vs_V1kpc.png'%self.dataset)
        plt.close()


plot = plot_Vmax_vs_V1kpc() # dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64) 
plot.plot()
