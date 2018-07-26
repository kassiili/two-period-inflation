import os, sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class plot_slice:

    def __init__(self, part_type, slice_axis, width, dataset='V1_LR_fix_127_z000p000', nfiles=16):

        self.dataset = dataset
        self.nfiles = nfiles
        self.reader = read_data.read_data(dataset=self.dataset, nfiles=self.nfiles)

        self.part_type = part_type
        self.slice_axis = slice_axis
        self.a, self.h, mass, self.boxsize = self.reader.read_header() 
        self.width = width
        self.coords = self.reader.read_dataset(self.part_type, 'Coordinates') * u.cm.to(u.Mpc)

        self.cm = self.calcCM()

        self.compute_slice()

    #Calculate center of mass for the dm particles:
    def calcCM(self):

        if (self.part_type != 1):
            dmCoords = self.reader.read_dataset(1, 'Coordinates') * u.cm.to(u.Mpc)
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

        gns = self.reader.read_subhaloData("GroupNumber")
        sgns = self.reader.read_subhaloData("SubGroupNumber")
        COPs = self.reader.read_subhaloData("CentreOfPotential")

        mask = np.logical_or(np.logical_and(gns == 1, sgns == 0), np.logical_and(gns == 2, sgns == 0))
        centrals = COPs[mask] * u.cm.to(u.Mpc)
        axes.scatter(centrals[:,x], centrals[:,y], s=10, c='red', edgecolor='none')

        print(self.cm[self.slice_axis], ', ', centrals[:,self.slice_axis])

#        axes.set_title('Particles (type %i) in a volume slice centered on a LG analogue'%self.part_type)

#        plt.show()
        plt.savefig('../Figures/V1_MR_mock_1_fix_082_z001p941/slice_partType%i.png'%self.part_type) 
        plt.close()

slice = plot_slice(1, 2, 0.03, dataset='V1_MR_mock_1_fix_082_z001p941', nfiles=1)
slice.plot()

#slice = plot_slice(1, 2, 0.03) #, dataset='snapshots/V1_MR_mock_fix/snapshot_082_z001p941'
#slice.plot()
#part_types = [0,1,4]
#for n in part_types:
#    slice = plot_slice(n, dataset='MR') 
#    slice.plot()

