import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from read_dataset import read_dataset
from read_header import read_header

class plot_slice:

    #Calculate center of mass:
    def calcCM(self):
        self.cm = [0, 0, 0]
        for i in range(self.coords[:,0].size): 
            self.cm += self.massTable[self.part_type] * self.coords[i]
        
        self.cm = self.cm / (self.massTable[self.part_type] * self.coords[:,0].size)


    #Divide particles in group 1 by subgroup:
    def divideIntoSubgroups_shitty(self):
        self.subgroups = dict()
        for i in range(self.coords[:,0].size):
            #Choose only particles in the slice that belong to group 1
            if (self.cm[0]-self.width/2 < self.coords[i,0] < self.cm[0]+self.width/2 and self.grpNumbers[i] == 1): 
                key = self.sgrpNumbers[i]
                if (key not in self.subgroups):
                    self.subgroups[key] = self.coords[i]
                else:
                    self.subgroups[key] = np.vstack([self.subgroups.get(key), self.coords[i]])


    #Divide particles in group 1 by subgroup:
    def divideIntoSubgroups_2(self):
        self.subgroups = dict()
        for i in range(self.coords[:,0].size):
            #Choose only particles in the slice that belong to group 1
            if (self.cm[0]-self.width/2 < self.coords[i,0] < self.cm[0]+self.width/2 and self.grpNumbers[i] == 1): 
                key = self.sgrpNumbers[i]
                if (key not in self.subgroups):
                    self.subgroups[key] = []
                self.subgroups.get(key).extend(self.coords[i].tolist())

        for subgroup in self.subgroups.values():
            subgroup = np.reshape(subgroup, newshape=(len(subgroup)/3,3)) #error: something is interpeted as float?

    #Divide particles in group 1 by subgroup:
    def divideIntoSubgroups(self):
        self.subgroups = dict()
        

        for subgroup in self.subgroups.values():
            subgroup = np.reshape(subgroup, newshape=(len(subgroup)/3,3)) #error: something is interpeted as float?


    def __init__(self):
        self.a, self.h, self.massTable, self.boxsize = read_header() 
        self.part_type = 1
        self.width = 1
        self.coords = read_dataset(self.part_type, 'Coordinates')
        self.sgrpNumbers = read_dataset(self.part_type, 'SubGroupNumber')
        self.grpNumbers = read_dataset(self.part_type, 'GroupNumber')

        self.calcCM()
        self.divideIntoSubgroups()


#    def compute_slice(self):
#        partsInSlice = []
#
#        for part in self.coords:
#            if (self.cm[0]-self.width/2 < part[0] < self.cm[0]+self.width/2):
#                partsInSlice.append(part)
#        return np.asarray(partsInSlice)

    def plot(self):
        plt.figure()
        axes = plt.gca() 
        axes.set_xlim([self.cm[1]-5, self.cm[1]+5])
        axes.set_ylim([self.cm[2]-5, self.cm[2]+5])

        #Get a list of all available colors (however, now the colors are arbitrary...)
        colors = list(clrs.cnames.keys())
        color = 0
        for subgroup in self.subgroups.values():
            #Do not include subgroups with only one particle (should that even be possible)
            if (len(subgroup.shape) > 1):
                axes.scatter(subgroup[:,1], subgroup[:,2], s=1, c=colors[color], edgecolor='none')
                color += 1

        plt.show()


slice = plot_slice() 
#slice.plot()
#slice.compute_slice()

