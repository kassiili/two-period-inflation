import sys
import numpy as np
import pandas as pd
import h5py
import time
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from read_subhaloData2_LR import read_subhaloData
from read_header_LR import read_header

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/LR_Plots/PlotPartPos/')
from read_dataset2 import read_dataset


class plot_Vmax_vs_V1kpc:


    def calcVelocitiesAt1kpc(self):
        # Sort by group numbers then by subgroup numbers:
        #sorting_idxs = np.lexsort((self.subGroupNumbers_fromPartData, self.groupNumbers_fromPartData))

        V1kpc = np.zeros((self.maxVelocities.size,))
        gravConst = 1.989/3.0857*10**15 * 6.674*10**(-11)    # m^3/(kg*s^2) -> kpc/(10^10 Msun) * (km/s)^2

        # Iterate through particle types:
        for partType in [0,1,4]:

            partsWithinR1kpc = np.zeros((self.maxVelocities.size,))
            masses = np.zeros((1))
            coords = np.zeros((1))

            # Iterate through subhaloes:
            for idx, subGroupNumber in enumerate(self.subGroupNumbers_fromSubhaloData):

                # Select coords of the particular subhalo:
                mask1 = (self.groupNumbers_fromPartData[partType] == self.groupNumbers_fromSubhaloData[idx])
                coords = self.coordsAndMasses[partType][mask1,:]
                help = self.subGroupNumbers_fromPartData[partType][mask1]
                mask2 = (help == subGroupNumber)
                coords = coords[mask2,:]
                
                coords = self.coordsAndMasses[partType][:,:3]
                maskR1kpc = np.sum((coords - self.cops[idx])**2, axis=1) < 10**(-6)     # Find the coordinates, whose distance from cop is less than 1kpc (unit of coords is Mpc), calculate how many there are. (np.sum(...) returns an array of distances, one for each vector in coords)
                if (partType == 1):
                    V1kpc[idx] += self.massTable[partType] * maskR1kpc.sum()
                else:
                    masses = self.coordsAndMasses[partType][:,3]
                    V1kpc[idx] += masses[maskR1kpc].sum()    # Need the same elements from masses as I need from coords..

                # Do not include subhaloes with less than 10 particles:
#                if (partsWithinR1kpc[idx] < 10):    
#                    partsWithinR1kpc[idx] = 0

        return np.sqrt(V1kpc * gravConst)

    def __init__(self):
        self.a, self.h, self.massTable, self.boxsize = read_header() 

        # Read particle data of all particle types:
        self.coordsAndMasses = 5*[0]
        self.subGroupNumbers_fromPartData = 5*[0]
        self.groupNumbers_fromPartData = 5*[0]
        for partType in [0,1,4]:
            # For particles other than dm (which all have the same mass), add particle mass to the end of the coord vector:
            if (partType == 1):
                self.coordsAndMasses[partType] = read_dataset(partType, 'Coordinates')
            else:
                coords = read_dataset(partType, 'Coordinates')
                masses = read_dataset(partType, 'Masses').reshape((coords[:,0].size, 1))

                self.coordsAndMasses[partType] = np.hstack((coords, masses))
            self.subGroupNumbers_fromPartData[partType] = read_dataset(partType, 'SubGroupNumber').astype(dtype="int16")
            self.groupNumbers_fromPartData[partType] = read_dataset(partType, 'GroupNumber').astype(dtype="int16")

        self.subGroupNumbers_fromSubhaloData = read_subhaloData('SubGroupNumber').astype(dtype="int16")
        self.groupNumbers_fromSubhaloData = read_subhaloData('GroupNumber').astype(dtype="int16")
        print(type(self.subGroupNumbers_fromSubhaloData[0]))

        self.maxVelocities = read_subhaloData('Vmax')
        self.cops = read_subhaloData('CentreOfPotential')

        start = time.clock()
        velocitiesAt1kpc = self.calcVelocitiesAt1kpc()
        print(time.clock() - start)

        mask = velocitiesAt1kpc > self.maxVelocities
        oddOnes = {}
        oddOnes['SGN'] = self.subGroupNumbers_fromSubhaloData[mask]
        oddOnes['GN'] = self.groupNumbers_fromSubhaloData[mask]

        for gn, sgn in zip(oddOnes['GN'], oddOnes['SGN']):
            print(gn, sgn)

        maskSat = np.logical_and(self.maxVelocities > 0, self.subGroupNumbers_fromSubhaloData != 0)
        maskIsol = np.logical_and(self.maxVelocities > 0, self.subGroupNumbers_fromSubhaloData == 0)
        self.maxVelocitiesSat = self.maxVelocities[maskSat]
        self.velocitiesAt1kpcSat = velocitiesAt1kpc[maskSat]
        self.maxVelocitiesIsol = self.maxVelocities[maskIsol]
        self.velocitiesAt1kpcIsol = velocitiesAt1kpc[maskIsol]

 
    def read_galaxies(self, part_type):
        """ """

        data = {}

        data['GroupNumber']  = read_dataset(part_type, 'GroupNumber')
        data['SubGroupNumber'] = read_dataset(part_type, 'SubGroupNumber')

        if itype == 1:
            data['mass'] = read_dataset_dm_mass()[mask] * u.g.to(u.Msun)
        else:
            data['mass'] = read_dataset(itype, 'Masses')[mask] * u.g.to(u.Msun)
        data['coords'] = read_dataset(itype, 'Coordinates')[mask] * u.cm.to(u.Mpc)

        # Periodic wrap coordinates around centre.
        boxsize = self.boxsize/self.h
        data['coords'] = np.mod(data['coords']-self.centre+0.5*boxsize,boxsize)+self.centre-0.5*boxsize

        return data


    def plot(self):
        fig = plt.figure()
        axes = plt.gca()

        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.scatter(self.velocitiesAt1kpcSat[self.velocitiesAt1kpcSat > 10], self.maxVelocitiesSat[self.velocitiesAt1kpcSat > 10], s=3, c='red', edgecolor='none', label='satellite galaxies')
        axes.scatter(self.velocitiesAt1kpcIsol[self.velocitiesAt1kpcIsol > 10], self.maxVelocitiesIsol[self.velocitiesAt1kpcIsol > 10], s=3, c='blue', edgecolor='none', label='isolated galaxies')

#        start = time.clock()
#        median = self.calc_median_trend2(self.maxVelocitiesSat, self.stellarMassesSat)
#        axes.plot(median[0], median[1], c='red', linestyle='--')

        x = np.arange(10,100)
        y = np.arange(10,100)
        axes.plot(x, y, c='black')

        axes.legend()
        axes.set_xlabel('$v_{1kpc}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$v_{max}[\mathrm{km s^{-1}}]$')

        plt.show()
#        plt.savefig('SMF_vs_Vmax.png')
#        plt.close()


plot = plot_Vmax_vs_V1kpc() 
plot.plot()
