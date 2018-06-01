import sys
import numpy as np
import pandas as pd
import h5py
import time
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from read_subhaloData_LR import read_subhaloData
from read_header_LR import read_header

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/LR_Plots/PlotPartPos/')
from read_dataset import read_dataset


class plot_Vmax_vs_V1kpc:

    # Divide particles by subgroup:
    def divideIntoSubgroups(self):
        # Get subgroup numbers from particle data:
        subGroupNumbers_fromPartData = read_dataset(self.part_type, 'SubGroupNumber')

        coordsBySubGroup = dict()
        for subGroupNumber in pd.unique(subGroupNumbers_fromPartData):
            coordsBySubGroup[subGroupNumber] = self.coords[subGroupNumbers_fromPartData == subGroupNumber,:] 

#        print('subgroups: ', len(coordsBySubGroup)) #60
#        print(self.subGroupNumbers_fromSubhaloData.size)            #1150
            
        return coordsBySubGroup


    def calcVelocitiesAt1kpc(self):
        partsWithinR1kpc = np.empty((self.maxVelocities.size,))
        gravConst = 1.989/3.0857*10**15 * 6.674*10**(-11)    # m^3/(kg*s^2) -> kpc/(10^10 Msol) * (km/s)^2

        sum = 0

        for idx, subGroupNumber in enumerate(self.subGroupNumbers_fromSubhaloData):
            cop = self.cops[idx]
            #cop = self.cops[self.subGroupNumbers_fromSubhaloData == subGroupNumber,:][0,:]    # Find center of potential (np.where(...) returns an array of indeces, which fulfill the condition – we only need the first element, since they all correspond to the same subhalo)
            coords = self.coordsBySubGroup[subGroupNumber]

            start = time.clock()
            partsWithinR1kpc[idx] = (np.sum((coords - cop)**2, axis=1) < 10**(-6)).sum()         # Find the coordinates, whose distance from cop is less than 1kpc (unit of coords is Mpc), calculate how many there are. (np.sum(...) returns an array of distances, one for each vector in coords)
            sum += time.clock()-start

            #print(coords.shape, ', ', cop.shape)

        print(sum) # Calculating the elements of the partsWithinR1kpc takes up most of the running time

        return np.sqrt(self.massTable[self.part_type] * partsWithinR1kpc * gravConst)


    def __init__(self):
        self.part_type = 1 
        self.a, self.h, self.massTable, self.boxsize = read_header() 
        self.width = 1 #Mpc/h
        self.coords = read_dataset(self.part_type, 'Coordinates')
        self.subGroupNumbers_fromSubhaloData = read_subhaloData('SubGroupNumber')
        self.maxVelocities = read_subhaloData('Vmax')
        self.cops = read_subhaloData('CentreOfPotential')

        self.coordsBySubGroup = self.divideIntoSubgroups()

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

        axes.scatter(self.velocitiesAt1kpcSat, self.maxVelocitiesSat, s=3, c='red', edgecolor='none', label='satellite galaxies')
        axes.scatter(self.velocitiesAt1kpcIsol, self.maxVelocitiesIsol, s=3, c='blue', edgecolor='none', label='isolated galaxies')

#        start = time.clock()
#        median = self.calc_median_trend2(self.maxVelocitiesSat, self.stellarMassesSat)
#        axes.plot(median[0], median[1], c='red', linestyle='--')

        axes.legend()
        axes.set_xlabel('$v_{1kpc}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$v_{max}[\mathrm{km s^{-1}}]$')

        axes.set_xlim(10, 100)
        axes.set_ylim(10, 150)

        plt.show()
#        plt.savefig('SMF_vs_Vmax.png')
#        plt.close()

plot = plot_Vmax_vs_V1kpc() 
plot.plot()
