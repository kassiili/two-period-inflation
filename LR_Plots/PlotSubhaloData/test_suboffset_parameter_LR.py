import numpy as np
import sys
import h5py
import time
import matplotlib.pyplot as plt
from read_subhaloData_LR import read_subhaloData
from read_partIDs_LR import read_partIDs
from read_header_LR import read_header

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/LR_Plots/PlotPartPos/')
from read_dataset import read_dataset

class plot_Rmax_vs_Vmax:

    def read_partIDs_bySubGroup(self):
        subOffsets = read_subhaloData('SubOffset')
        subLengthsType = read_subhaloData('SubLengthType')
        partIDs = read_partIDs('ParticleID')
        subGroupNumbers = read_subhaloData('SubGroupNumber')

        print(subOffsets.size, ', ', subGroupNumbers.size)
        partIDs_bySubGroup = dict()
        print(max(subOffsets))
        print(partIDs.size)
        print(self.maxVelocities.size)
        print(subLengthsType.sum())
        
#        for idx, subOffset in enumerate(subOffsets):
#            partIDs_bySubGroup[subGroupNumbers[idx]] = 

        # If I want to iterate through subgroup numbers I'll (or at least so it seems) have to also iterate through all the index slices
        # that contain particles of that subgroup. This is because the slices can be of varying lengths, which makes selecting elements 
        # of the particle ID array by an index array very difficult. Thus, it might be easier to iterate through the slices first.
        # The problem with that approach is that I'll need to append the particle ID array on a large number of occations.

        # Subgroups are UNIQUELY represented in the subhalo catalogues! The reason the same subgroup numbers repeat is that the same 
        # numbers are used in different (FoF) GROUPS.

#        for subGroupNumber in enumerate(np.unique(subGroupNumbers)):
#            startIdxs = subOffsets[subGroupNumbers == subGroupNumber]
#            endIdxs = startIdxs + subLengthsType[subGroupNumbers == subGroupNumber, self.part_type]
#            print(subGroupNumber)
#            print(startIdxs.size)
#            print(max(startIdxs))
#
#            indices = []
#            for i in range(startIdxs.size):
#                #print(startIdxs[i])
#                indices.extend(range(startIdxs[i], endIdxs[i]))
#            #print(indices)
#            partIDs_bySubGroup[subGroupNumber] = np.take(partIDs, indices)


    def __init__(self):
        #self.a, self.h, mass, self.boxsize = read_header() 
        self.part_type = 1
        self.maxVelocities = read_subhaloData('Vmax')
        #self.stellarMasses = read_subhaloData('Stars/Mass')
        
        self.coords = read_dataset(self.part_type, 'Coordinates')

        start = time.clock()
        self.read_partIDs_bySubGroup()
        print(time.clock()-start)
    
#        maskSat = np.logical_and.reduce((maxVelocities > 0, maxRadii > 0, self.subGroupNumbers != 0, self.stellarMasses > 0))
#        maskIsol = np.logical_and.reduce((maxVelocities > 0, maxRadii > 0, self.subGroupNumbers == 0, self.stellarMasses > 0))
#
#        self.maxVelocitiesSat = maxVelocities[maskSat]
#        self.maxRadiiSat = maxRadii[maskSat]
#        self.maxVelocitiesIsol = maxVelocities[maskIsol]
#        self.maxRadiiIsol = maxRadii[maskIsol]


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
#        axes.set_xlim([10, 100])
#        axes.set_ylim([10**6, 10**10])

        plt.savefig('rmax_vs_vmax.png')
        plt.close()

slice = plot_Rmax_vs_Vmax()
#slice.plot()
