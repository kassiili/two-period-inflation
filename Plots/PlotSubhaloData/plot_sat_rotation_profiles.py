import sys
import h5py
import numpy as np
from collections import Counter
import time
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 
from read_subhaloData import read_subhaloData

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/PlotPartPos/')
from read_dataset import read_dataset
from read_dataset_dm_mass import read_dataset_dm_mass

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

class SatRotationCurves:

    def __init__(self, gn, dataset='LR'):

        self.gn = gn
        self.dataset = dataset
        self.a, self.h, self.massTable, self.boxsize = read_header(dataset=self.dataset)
        self.boxsize = self.boxsize*1000/self.h # Mpc/h -> kpc

        self.sat_data, self.sat_cnt = self.read_satellite_data()
        print('satellites: ', self.sat_cnt)
        self.read_particles()

        # Plot.
        self.plot()

    def read_satellite_data(self):

        sat_data = {}
        sat_data['GNs'] = read_subhaloData('GroupNumber', dataset=self.dataset)
        sat_data['SGNs'] = read_subhaloData('SubGroupNumber', dataset=self.dataset)
        sat_data['vmax'] = read_subhaloData('Vmax', dataset=self.dataset)/100000  # cm/s to km/s
        sat_data['COPs'] = read_subhaloData('CentreOfPotential', dataset=self.dataset) * u.cm.to(u.kpc)

        halo_centre = sat_data['COPs'][np.logical_and(sat_data['GNs']==self.gn, sat_data['SGNs']==0)]
        dists_to_centre = np.linalg.norm(sat_data['COPs'] - halo_centre, axis=1)
        
        # Choose satellites (by definition d < 300 kpc) with vmax > 12 km/s
        sat_mask = np.logical_and.reduce((sat_data['GNs']==self.gn, sat_data['SGNs']!=0, dists_to_centre<300, sat_data['vmax']>12))
        for attr in sat_data.keys():
            sat_data[attr] = sat_data[attr][sat_mask]

        # Sort by sgn to ease particle data handling:
        sort_mask = np.argsort(sat_data['SGNs'])
        for attr in sat_data.keys():
            sat_data[attr] = sat_data[attr][sort_mask]

        return sat_data, sat_data['SGNs'].size

    def read_particles(self):

        data = dict.fromkeys(['GroupNumber', 'SubGroupNumber', 'Coordinates', 'Masses'])

        # Periodic wrap coordinates around centre.
        #data['coords'] = np.mod(data['coords']-self.centre+0.5*self.boxsize,self.boxsize)+self.centre-0.5*self.boxsize

        # Combine into arrays by attribute:
        for attr in data.keys():
            print(attr)
            if attr == 'Coordinates':
                data[attr] = np.vstack((read_dataset(0, attr, dataset=self.dataset),
                    read_dataset(1, attr, dataset=self.dataset),
                    read_dataset(4, attr, dataset=self.dataset),
                    read_dataset(5, attr, dataset=self.dataset)))
            elif attr == 'Masses':
                data[attr] = np.concatenate((read_dataset(0, attr, dataset=self.dataset),
                    read_dataset_dm_mass(dataset=self.dataset),
                    read_dataset(4, attr, dataset=self.dataset),
                    read_dataset(5, attr, dataset=self.dataset)))
            else:
                data[attr] = np.concatenate((read_dataset(0, attr, dataset=self.dataset),
                    read_dataset(1, attr, dataset=self.dataset),
                    read_dataset(4, attr, dataset=self.dataset),
                    read_dataset(5, attr, dataset=self.dataset)))

        # Exclude particles not belonging to some of the halo's satellites: 
        mask_halo = np.logical_and(data['GroupNumber'] == self.gn, 
                np.isin(data['SubGroupNumber'], self.sat_data['SGNs']))
        for attr in data.keys():
            data[attr] = data[attr][mask_halo]

        # Sort by sgn:
        sort_mask = np.argsort(data['SubGroupNumber'])
        self.sat_data['coords'] = data['Coordinates'][sort_mask] * u.cm.to(u.kpc)
        self.sat_data['masses'] = data['Masses'][sort_mask] * u.g.to(u.Msun)

        # Find offsets in the coords and masses arrays for each satellite (Counter returns a dict object. A value behind a certain key gives the number of instances of the key in argument array):
        part_cnt = Counter(data['SubGroupNumber']).values()
        print(Counter(data['SubGroupNumber']))

        # Convert from dict_values to ndarray:
        part_cnt = np.asarray(list(part_cnt))

        print('satellites: ', part_cnt.size)

        # Zero must be added to the beginning of the array, since the offset of the first satellite must be zero:
        self.sat_data['offsets'] = np.cumsum(np.insert(part_cnt, 0, 0))

    def compute_rotation_curve(self, sat_idx):
        """ Compute the rotation curve of given satellite. """

        start = self.sat_data['offsets'][sat_idx]
                                    #  == sat_idx + 1
        end = self.sat_data['offsets'][sat_idx-self.sat_cnt+1] # Handles special case of last item
        print(end-start)

        # Compute distance to centre.
        r = np.linalg.norm(self.sat_data['coords'][start:end] - self.sat_data['COPs'][sat_idx], axis=1)

        mask = np.argsort(r)
        r = r[mask]

        cmass = np.cumsum(self.sat_data['masses'][start:end][mask])

        # Begin rotation curve from the 10th particle to reduce noise at the low end of the curve.
        r = r[10:]
        cmass = cmass[10:]

        # Compute velocity.
        myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
        v = np.sqrt((myG * cmass) / r)

        # Return r in kpc and v in km/s.
        return r, v

    def plot(self):
        fig, axes = plt.subplots()

        for idx in range(self.sat_cnt):
            r, v = self.compute_rotation_curve(idx)
            axes.plot(r, v, c='black')

        # Save plot.
        plt.minorticks_on()
        #plt.title('Rotation curve of halo with GN = %i and SGN = %i (%s)'%(gn,sgn,self.dataset))
        plt.ylabel('Velocity [km/s]'); plt.xlabel('r [kpc]')
        plt.xlim(0, 80); # plt.tight_layout()

        #fig.savefig('Figures/RotationCurve_g%i-sg%i_%s.png'%(gn,sgn,self.dataset))
        plt.show()
        plt.close()

SatRotationCurves(1, dataset='LR')

