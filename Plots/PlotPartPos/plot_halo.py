import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from read_dataset import read_dataset

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/PlotSubhaloData')
from read_subhaloData import read_subhaloData

class plot_slice:

    def __init__(self, gn, sgn, slice_axis, dataset='LR'):
        self.dataset = dataset
        self.gn = gn
        self.sgn = sgn
        self.slice_axis = slice_axis
        self.a, self.h, self.massTable, self.boxsize = read_header(dataset=self.dataset) 
        self.width = 0.01 #Mpc

        # Load data.
        gas    = self.read_galaxy(0, gn, sgn)
        dm     = self.read_galaxy(1, gn, sgn)
        stars  = self.read_galaxy(4, gn, sgn)
        BHs    = self.read_galaxy(5, gn, sgn)

        self.coords = np.vstack((gas, dm, stars, BHs))

        #Calculate center of mass for dm particles:
        #self.centre = self.calcCM(read_dataset(1, 'Coordinates', dataset=self.dataset) * u.cm.to(u.Mpc))

        gns = read_subhaloData("GroupNumber", dataset=self.dataset)
        sgns = read_subhaloData("SubGroupNumber", dataset=self.dataset)
        mask = np.logical_and(gns == gn, sgns == sgn)
        self.centre = read_subhaloData("CentreOfPotential", dataset=self.dataset)[mask].reshape((3)) * u.cm.to(u.Mpc)

        self.compute_slice()

    def read_galaxy(self, itype, gn, sgn):

        # Load data, then mask to selected GroupNumber and SubGroupNumber.
        gns  = read_dataset(itype, 'GroupNumber', dataset=self.dataset)
        sgns = read_dataset(itype, 'SubGroupNumber', dataset=self.dataset)

        mask = np.logical_and(gns == gn, sgns == sgn)
        return read_dataset(itype, 'Coordinates', dataset=self.dataset)[mask] * u.cm.to(u.Mpc)

    #Calculate centre of mass:
    def calcCM(self, dmCoords):
        cm = np.mean(dmCoords, axis=0)
        return cm

    def compute_slice(self):

        #Volume limits (in the direction of the zeroth coordinate):
        min0 = self.centre[self.slice_axis]-self.width/2
        max0 = self.centre[self.slice_axis]+self.width/2

        mask = np.logical_and(min0 < self.coords[:,self.slice_axis], 
            self.coords[:,self.slice_axis] < max0)
            
        self.coords = self.coords[mask,:] 

    def plot(self):
        fig = plt.figure()
        axes = plt.gca() 

        help = [0,1,2]
        x=help[self.slice_axis-2]; y=help[self.slice_axis-1]

        axes.scatter(self.coords[:,x], self.coords[:,y], s=0.01)
        axes.scatter(self.centre[x], self.centre[y], s=5, edgecolor='none', c='red')

        plt.title('Slice plot of halo GN = %i and SGN = %i (%s)'%(self.gn,self.sgn,self.dataset))

        plt.show()
#        plt.savefig('dm_bySubgroup_%s.png'%self.dataset)
#        plt.close()

slice = plot_slice(1, 0, 1, dataset='MR') 
slice.plot()

