import sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from read_dataset import read_dataset

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

class plot_slice:

    #Calculate center of mass for the dm particles:
    def calcCM(self, dmCoords):
        cm = np.mean(dmCoords, axis=0)
        return cm

    def compute_slice(self):
        min0 = self.cm[0]-self.width/2
        max0 = self.cm[0]+self.width/2
        self.coords = self.coords[np.logical_and(min0 < self.coords[:,0], self.coords[:,0] < max0),:] 

    def __init__(self, part_type, dataset='LR'):
        self.dataset = dataset
        self.a, self.h, mass, self.boxsize = read_header() 
        self.part_type = part_type
        self.width = 0.2
        self.coords = read_dataset(self.part_type, 'Coordinates') * u.cm.to(u.Mpc)

        #Calculate center of mass:
        if (self.part_type != 1):
            self.cm = self.calcCM(read_dataset(1,'Coordinates') * u.cm.to(u.Mpc))
        else:
            self.cm = self.calcCM(self.coords)

        self.compute_slice()

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

slice = plot_slice(1)
slice.plot()
#part_types = [0,1,4]
#for n in part_types:
#    print(n)
#    slice = plot_slice(n) 
#    slice.plot()
