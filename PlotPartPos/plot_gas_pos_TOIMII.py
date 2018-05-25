import numpy as np
import h5py
import matplotlib.pyplot as plt
from read_dataset import read_dataset
from read_header import read_header

class plot_slice:

    def __init__(self):
        self.a, self.h, massTable, self.boxsize = read_header() 
        part_type = 0
        self.width = 0.1
        self.coords = read_dataset(part_type, 'Coordinates')
        self.dm = read_dataset(1, 'Coordinates')

        #Calculate center of mass:
        self.cm = [0, 0, 0]
        for i in range(self.dm[:,0].size): #
            self.cm += massTable[1] * self.dm[i]
 
        self.cm = self.cm / (massTable[1] * self.dm[:,0].size)


    def compute_slice(self):
        partsInSlice = []

        for part in self.coords:
            if (self.cm[0]-self.width/2 < part[0] < self.cm[0]+self.width/2): 
                partsInSlice.append(part)
        return np.asarray(partsInSlice)

    def plot(self):
        plt.figure()
        axes = plt.gca() 
        axes.set_xlim([self.cm[1]-5, self.cm[1]+5])
        axes.set_ylim([self.cm[2]-5, self.cm[2]+5])

        parts = self.compute_slice()
        plt.scatter(parts[:,1], parts[:,2], s=0.01)

        plt.savefig('gas_slice.png')
        plt.close()

slice = plot_slice() 
slice.plot()
#slice.compute_slice()

