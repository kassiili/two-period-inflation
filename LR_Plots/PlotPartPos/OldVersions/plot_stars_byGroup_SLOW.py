import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import time
from read_dataset import read_dataset
from read_header import read_header

class plot_slice:

    #Calculate center of mass:
    def calcCM(self, dmCoords):
        cm = np.mean(dmCoords, axis=0)
        return cm


    #Divide particles in group 1 by subgroup:
    def divideIntoSubgroups(self):
        subgroups = dict()
        min0 = self.cm[0]-self.width/2
        max0 = self.cm[0]+self.width/2
        for i in range(self.coords[:,0].size):
            #Choose only particles in the slice that belong to group 1
            if (self.groupNumbers[i] == 1 and min0 < self.coords[i,0] < max0): 
                key = self.subGroupNumbers[i]
                if (key not in subgroups):
                    subgroups[key] = self.coords[i]
                else:
                    subgroups[key] = np.vstack([subgroups.get(key), self.coords[i]])
#                if (key not in subgroups):
#                    subgroups[key] = []
#                subgroups.get(key).extend(self.coords[i].tolist())
                                                             #THIS is probably quite slow too..

        return subgroups


    def __init__(self):
        self.a, self.h, self.massTable, self.boxsize = read_header() 
        part_type = 4
        self.width = 1 #Mpc/h
        self.coords = read_dataset(part_type, 'Coordinates')
        self.subGroupNumbers = read_dataset(part_type, 'SubGroupNumber')
        self.groupNumbers = read_dataset(part_type, 'GroupNumber')

        #Calculate center of mass for dm particles:
        self.cm = self.calcCM(read_dataset(1, 'Coordinates'))

        self.coordsBySubGroup = self.divideIntoSubgroups()


    def plot(self):
        fig = plt.figure()
        axes = plt.gca() 
        axes.set_xlim([self.cm[1]-5, self.cm[1]+5])
        axes.set_ylim([self.cm[2]-5, self.cm[2]+5])

        #Get a list of all available colors (however, now the colors are arbitrary...)
        colors = list(clrs.cnames.keys())
        color = 0
        for subgroup in self.coordsBySubGroup.values():
            #Do not include subgroups with only one particle (should that even be possible)
            if (len(subgroup.shape) > 1):
                axes.scatter(subgroup[:,1], subgroup[:,2], s=0.1, c=colors[color], edgecolor='none')
                color += 1

        plt.show()
#        plt.savefig('stars_slice_bySGroup.png')
#        plt.close()

slice = plot_slice() 
slice.plot()

