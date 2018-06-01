import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from read_dataset_MR import read_dataset
from read_header_MR import read_header

class plot_slice:

    #Calculate center of mass:
    def calcCM(self, dmCoords):
        cm = np.mean(dmCoords, axis=0)
        return cm


    #Divide particles in group 1 by subgroup:
    def divideIntoSubgroups(self):

        #Volume limits (in the direction of the zeroth coordinate):
        min0 = self.cm[0]-self.width/2
        max0 = self.cm[0]+self.width/2

        coordsBySubGroup = dict()
        for subGroupNumber in np.unique(self.subGroupNumbers):
            mask = np.logical_and.reduce((self.groupNumbers == 1,
                self.subGroupNumbers == subGroupNumber,
                min0 < self.coords[:,0], 
                self.coords[:,0] < max0))
            
            coordsBySubGroup[subGroupNumber] = self.coords[mask,:] 
            
        return coordsBySubGroup


    def __init__(self, part_type):
        self.part_type = part_type
        self.a, self.h, self.massTable, self.boxsize = read_header() 
        self.width = 0.1 #Mpc/h
        self.coords = read_dataset(part_type, 'Coordinates')
        self.subGroupNumbers = read_dataset(part_type, 'SubGroupNumber')
        self.groupNumbers = read_dataset(part_type, 'GroupNumber')

        #Calculate center of mass for dm particles:
        self.cm = self.calcCM(read_dataset(1, 'Coordinates'))

        self.coordsBySubGroup = self.divideIntoSubgroups()


    def plot(self):
        fig = plt.figure()
        axes = plt.gca() 
        axes.set_xlim([self.cm[1]-0.3, self.cm[1]+0.3])
        axes.set_ylim([self.cm[2]-0.5, self.cm[2]+0.1])

        col = []
        x = []
        y = []
        i = 1
        for coords in self.coordsBySubGroup.values():
            col.extend([i] * coords[:,0].size)
            x.extend(coords[:,1].tolist())
            y.extend(coords[:,2].tolist())
            i+=1

        axes.scatter(x, y, s=0.1, c=col, edgecolor='none', cmap='viridis_r')

        plt.savefig('slice_partType%i_bySubGroup_MR.png'%self.part_type)
        plt.close()

slice = plot_slice(1) 
slice.plot()

