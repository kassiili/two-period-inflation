import os, sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class plot_slice:

    def __init__(self, gn, sgn, slice_axis, dataset='snapshots/V1_LR_fix_127_z000p000', nfiles=16):
        self.dataset = dataset
        self.nfiles = nfiles
        self.reader = read_data.read_data(dataset=self.dataset, nfiles=self.nfiles)

        self.gn = gn
        self.sgn = sgn
        self.slice_axis = slice_axis
        self.a, self.h, self.massTable, self.boxsize = self.reader.read_header() 
        self.width = 0.01 #Mpc

        # Load data.
        gas    = self.read_galaxy(0, gn, sgn)
        dm     = self.read_galaxy(1, gn, sgn)
        stars  = self.read_galaxy(4, gn, sgn)
        BHs    = self.read_galaxy(5, gn, sgn)

        self.coords = np.vstack((gas, dm, stars, BHs))

        #Calculate center of mass for dm particles:
        #self.centre = self.calcCM(self.reader.read_dataset(1, 'Coordinates', dataset=self.dataset) * u.cm.to(u.Mpc))

        gns = self.reader.read_subhaloData("GroupNumber")
        sgns = self.reader.read_subhaloData("SubGroupNumber")
        mask = np.logical_and(gns == gn, sgns == sgn)
        self.centre = self.reader.read_subhaloData("CentreOfPotential")[mask].reshape((3)) * u.cm.to(u.Mpc)

        self.compute_slice()

    def read_galaxy(self, itype, gn, sgn):

        # Load data, then mask to selected GroupNumber and SubGroupNumber.
        gns  = self.reader.read_dataset(itype, 'GroupNumber')
        sgns = self.reader.read_dataset(itype, 'SubGroupNumber')

        mask = np.logical_and(gns == gn, sgns == sgn)
        return self.reader.read_dataset(itype, 'Coordinates')[mask] * u.cm.to(u.Mpc)

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

slice = plot_slice(1, 0, 1) 
slice.plot()

