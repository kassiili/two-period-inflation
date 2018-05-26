import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from read_dataset import read_dataset
from read_header import read_header

class plot_slice:

    def __init__(self):
        self.a, self.h, mass, self.boxsize = read_header() 
        part_type = 1
        self.width = 1
        self.coords = read_dataset(part_type, 'Coordinates')
        self.sgrpNumbers = read_dataset(part_type, 'SubGroupNumber')
        self.grpNumbers = read_dataset(part_type, 'GroupNumber')

        #Calculate center of mass:
        self.cm = [0, 0, 0]
        for i in range(self.coords[:,0].size): 
            self.cm += mass[part_type] * self.coords[i]
        
        self.cm = self.cm / (mass[part_type] * self.coords[:,0].size)

        #Divide particles in group 1 by subgroup:
        self.subgroups = dict()
        for i in range(self.coords[:,0].size):
            #Choose only particles in the slice that belong to group 1
            if (self.cm[0]-self.width/2 < self.coords[i,0] < self.cm[0]+self.width/2 and self.grpNumbers[i] == 1): 
                key = self.sgrpNumbers[i]
                if (key not in self.subgroups):
                    self.subgroups[key] = self.coords[i]
                else:
                    self.subgroups[key] = np.vstack([self.subgroups.get(key), self.coords[i]])


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

        plt.savefig('dm_slice_bySGroup.png')
        plt.close()

slice = plot_slice() 
slice.plot()
#slice.compute_slice()

