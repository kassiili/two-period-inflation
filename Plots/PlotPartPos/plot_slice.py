import sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from read_dataset import read_dataset

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/PlotSubhaloData')
from read_subhaloData import read_subhaloData

class plot_slice:

    def __init__(self, part_type, slice_axis, dataset='LR'):
        self.part_type = part_type
        self.slice_axis = slice_axis
        self.dataset = dataset
        self.a, self.h, mass, self.boxsize = read_header(dataset=self.dataset) 
        self.width = 0.03
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
        min0 = self.cm[self.slice_axis]-self.width/2
        max0 = self.cm[self.slice_axis]+self.width/2
        self.coords = self.coords[np.logical_and(min0 < self.coords[:,self.slice_axis], self.coords[:,self.slice_axis] < max0),:] 

    def plot(self):
        plt.figure()
        axes = plt.gca() 

        help = [0,1,2]
        x=help[self.slice_axis-2]; y=help[self.slice_axis-1]
        axes.set_xlim([self.cm[x]-7, self.cm[x]+7])
        axes.set_ylim([self.cm[y]-7, self.cm[y]+7])

        axes.scatter(self.coords[:,x], self.coords[:,y], s=0.01)

        gns = read_subhaloData("GroupNumber", dataset=self.dataset)
        sgns = read_subhaloData("SubGroupNumber", dataset=self.dataset)
        COPs = read_subhaloData("CentreOfPotential", dataset=self.dataset)

        mask = np.logical_or(np.logical_and(gns == 1, sgns == 0), np.logical_and(gns == 2, sgns == 0))
        centrals = COPs[mask] * u.cm.to(u.Mpc)
        axes.scatter(centrals[:,x], centrals[:,y], s=10, c='red', edgecolor='none')

        print(self.cm[self.slice_axis], ', ', centrals[:,self.slice_axis])

#        axes.set_title('Particles (type %i) in a volume slice centered on a LG analogue'%self.part_type)

        plt.show()
#        plt.savefig('slice_partType%i%s.png'%(self.part_type, self.dataset)) 
#        plt.close()

slice = plot_slice(1, 2, dataset='MR')
slice.plot()
#part_types = [0,1,4]
#for n in part_types:
#    slice = plot_slice(n, dataset='MR') 
#    slice.plot()

