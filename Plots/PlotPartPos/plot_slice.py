import sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from read_dataset import read_dataset

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

class plot_slice:

    def __init__(self, part_type, dataset='LR'):
        self.part_type = part_type
        self.dataset = dataset
        self.a, self.h, mass, self.boxsize = read_header(dataset=self.dataset) 
        self.width = 0.2
        self.coords = read_dataset(self.part_type, 'Coordinates', dataset=self.dataset) * u.cm.to(u.Mpc)

        self.cm = self.calcCM()

        self.compute_slice()

    #Calculate center of mass for the dm particles:
    def calcCM(self):

        if (self.part_type != 1):
            dmCoords = read_dataset(1, 'Coordinates', dataset=self.dataset) * u.cm.to(u.Mpc)
        else:
            dmCoords = self.coords

        return np.mean(dmCoords, axis=0)

    def compute_slice(self):
        min0 = self.cm[0]-self.width/2
        max0 = self.cm[0]+self.width/2
        self.coords = self.coords[np.logical_and(min0 < self.coords[:,0], self.coords[:,0] < max0),:] 

    def plot(self):
        plt.figure()
        axes = plt.gca() 
        axes.set_xlim([self.cm[1]-5, self.cm[1]+5])
        axes.set_ylim([self.cm[2]-5, self.cm[2]+5])

        axes.scatter(self.coords[:,1], self.coords[:,2], s=0.01)

        axes.set_title('Particles (type %i) in a volume slice centered on a LG analogue'%self.part_type)

        plt.show()
#        plt.savefig('slice_partType%i%s.png'%(self.part_type, self.dataset)) 
#        plt.close()

#slice = plot_slice(4, dataset='LR')
#slice.plot()
part_types = [0,1,4]
for n in part_types:
    slice = plot_slice(n, dataset='MR') 
    slice.plot()

