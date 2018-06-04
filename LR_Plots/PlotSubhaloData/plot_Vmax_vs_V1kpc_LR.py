import sys
import numpy as np
import pandas as pd
import h5py
import time
import matplotlib.pyplot as plt
from read_subhaloData2_LR import read_subhaloData
from read_header_LR import read_header

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/LR_Plots/PlotPartPos/')
from read_dataset2 import read_dataset


class plot_Vmax_vs_V1kpc:


    def calcVelocitiesAt1kpc(self):
        # Get subgroup and group numbers from particle data:
        subGroupNumbers_fromPartData = read_dataset(self.part_type, 'SubGroupNumber')
        groupNumbers_fromPartData = read_dataset(self.part_type, 'GroupNumber')

        partsWithinR1kpc = np.empty((self.maxVelocities.size,))
        gravConst = 1.989/3.0857*10**15 * 6.674*10**(-11)    # m^3/(kg*s^2) -> kpc/(10^10 Msol) * (km/s)^2

        sum = 0
        sum1 = 0
        sum2 = 0

        # Iterate through subhaloes:
        for idx, subGroupNumber in enumerate(self.subGroupNumbers_fromSubhaloData):
            start = time.clock()
            mask = np.logical_and(subGroupNumbers_fromPartData == subGroupNumber, groupNumbers_fromPartData == self.groupNumbers_fromSubhaloData[idx])
            coords = self.coords[mask,:]    # Select coordinates from the particular subhalo.
            sum += time.clock() - start

            start2 = time.clock()
            partsWithinR1kpc[idx] = (np.sum((coords - self.cops[idx])**2, axis=1) < 10**(-6)).sum()     # Find the coordinates, whose distance from cop is less than 1kpc (unit of coords is Mpc), calculate how many there are. (np.sum(...) returns an array of distances, one for each vector in coords)
            sum2 += time.clock() - start

            # Do not include subhaloes with less than 10 particles inside 1kpc circular radius:
            if (partsWithinR1kpc[idx] < 10):    
                partsWithinR1kpc[idx] = 0

        print(sum) # this takes time!
        return np.sqrt(self.massTable[self.part_type] * partsWithinR1kpc * gravConst)


    def __init__(self):
        self.part_type = 1 
        self.a, self.h, self.massTable, self.boxsize = read_header() 
        self.coords = read_dataset(self.part_type, 'Coordinates')
        self.subGroupNumbers_fromSubhaloData = read_subhaloData('SubGroupNumber')
        self.groupNumbers_fromSubhaloData = read_subhaloData('GroupNumber')
        self.maxVelocities = read_subhaloData('Vmax')
        self.cops = read_subhaloData('CentreOfPotential')

        start = time.clock()
        velocitiesAt1kpc = self.calcVelocitiesAt1kpc()
        print(time.clock() - start)

        maskSat = np.logical_and(self.maxVelocities > 0, self.subGroupNumbers_fromSubhaloData != 0)
        maskIsol = np.logical_and(self.maxVelocities > 0, self.subGroupNumbers_fromSubhaloData == 0)
        self.maxVelocitiesSat = self.maxVelocities[maskSat]
        self.velocitiesAt1kpcSat = velocitiesAt1kpc[maskSat]
        self.maxVelocitiesIsol = self.maxVelocities[maskIsol]
        self.velocitiesAt1kpcIsol = velocitiesAt1kpc[maskIsol]


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
