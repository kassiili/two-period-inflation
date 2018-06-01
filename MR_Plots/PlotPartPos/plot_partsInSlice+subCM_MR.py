import numpy as np
import h5py
import matplotlib.pyplot as plt
from read_dataset_MR import read_dataset
from read_header_MR import read_header

import sys
sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/MR_Plots/PlotSubhaloData')
from read_subhaloData_MR import read_subhaloData

class plot_slice:


    #Calculate center of mass for the dm particles:
    def calcCM(self):
        if (self.part_type != 1):
            cm = np.mean(read_dataset(1,'Coordinates'), axis=0)
        else:
            cm = np.mean(self.coords, axis=0)
        return cm


    def compute_slice(self):
        min0 = self.cm[0]-self.width/2
        max0 = self.cm[0]+self.width/2
        self.coords = self.coords[np.logical_and(min0 < self.coords[:,0], self.coords[:,0] < max0),:] 
        self.cops = self.cops[np.logical_and(min0 < self.cops[:,0], self.cops[:,0] < max0),:] 
        self.coms = self.coms[np.logical_and(min0 < self.coms[:,0], self.coms[:,0] < max0),:] 


    def __init__(self, part_type):
        self.a, self.h, mass, self.boxsize = read_header() 
        self.part_type = part_type
        self.width = 0.2
        self.coords = read_dataset(self.part_type, 'Coordinates')
        self.cops = read_subhaloData('CentreOfPotential')
        self.coms = read_subhaloData('CentreOfMass')

        self.cm = self.calcCM()

        self.compute_slice()


    def plot(self):
        plt.figure()
        axes = plt.gca() 
        axes.set_xlim([self.cm[1]-5, self.cm[1]+5])
        axes.set_ylim([self.cm[2]-5, self.cm[2]+5])

        axes.scatter(self.coords[:,1], self.coords[:,2], s=0.01)
        axes.scatter(self.cops[:,1], self.cops[:,2], s=5, c='red', edgecolor='none')
        axes.scatter(self.coms[:,1], self.coms[:,2], s=5, c='blue', edgecolor='none')

        axes.set_title('Particles (type %i) in a volume slice'%self.part_type)

        plt.show()
        #plt.savefig('slice_partType%i_MR.png'%self.part_type)
        #plt.close()


slice = plot_slice(4)
slice.plot()
#part_types = [0,1,4]
#for n in part_types:
#    slice = plot_slice(n) 
#    slice.plot()

