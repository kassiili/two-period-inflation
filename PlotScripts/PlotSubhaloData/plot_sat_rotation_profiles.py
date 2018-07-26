import os, sys
import h5py
import numpy as np
from collections import Counter
import time
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class SatRotationCurves:

    def __init__(self, gn, dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192):

        self.dataset = dataset
        self.nfiles_part = nfiles_part
        self.nfiles_group = nfiles_group
        self.reader = read_data.read_data(dataset=self.dataset, nfiles_part=self.nfiles_part, nfiles_group=self.nfiles_group)

        self.gn = gn
        self.a, self.h, self.massTable, self.boxsize = self.reader.read_header()
        self.boxsize = self.boxsize*1000/self.h # Mpc/h -> kpc

        self.sat_data, self.sat_cnt = self.read_satellite_data()
        self.read_particles()

        # Plot.
        self.plot()

    def read_satellite_data(self):

        sat_data = {}
        sat_data['GNs'] = self.reader.read_subhaloData('GroupNumber')
        sat_data['SGNs'] = self.reader.read_subhaloData('SubGroupNumber')
        sat_data['vmax'] = self.reader.read_subhaloData('Vmax')/100000  # cm/s to km/s
        sat_data['COPs'] = self.reader.read_subhaloData('CentreOfPotential') * u.cm.to(u.kpc)

        halo_centre = sat_data['COPs'][np.logical_and(sat_data['GNs']==self.gn, sat_data['SGNs']==0)]
        dists_to_centre = np.linalg.norm(sat_data['COPs'] - halo_centre, axis=1)
        
        # Choose satellites (by definition d < 300 kpc) with vmax > 12 km/s
        sat_mask = np.logical_and.reduce((sat_data['GNs']==self.gn, sat_data['SGNs']!=0, 
            dists_to_centre<300, sat_data['vmax']>12))
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
            if attr == 'Coordinates':
                data[attr] = np.vstack((self.reader.read_dataset(0, attr),
                    self.reader.read_dataset(1, attr),
                    self.reader.read_dataset(4, attr),
                    self.reader.read_dataset(5, attr)))
            elif attr == 'Masses':
                data[attr] = np.concatenate((self.reader.read_dataset(0, attr),
                    self.reader.read_dataset_dm_mass(),
                    self.reader.read_dataset(4, attr),
                    self.reader.read_dataset(5, attr)))
            else:
                data[attr] = np.concatenate((self.reader.read_dataset(0, attr),
                    self.reader.read_dataset(1, attr),
                    self.reader.read_dataset(4, attr),
                    self.reader.read_dataset(5, attr)))

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

        # Convert from dict_values to ndarray:
        part_cnt = np.asarray(list(part_cnt))

        # Zero must be added to the beginning of the array, since the offset of the first satellite must be zero:
        self.sat_data['offsets'] = np.insert(np.cumsum(part_cnt[:-1]), 0, 0)

    def compute_rotation_curve(self, sat_idx):
        """ Compute the rotation curve of given satellite. """

        start = self.sat_data['offsets'][sat_idx]
                                    #  == sat_idx + 1
        end = self.sat_data['offsets'][sat_idx-self.sat_cnt+1] # Handles special case of last item

        # Compute distance to centre.
        if end==0:
            r = np.linalg.norm(self.sat_data['coords'][start:] - self.sat_data['COPs'][sat_idx], axis=1)
            mass = self.sat_data['masses'][start:]
        else:
            r = np.linalg.norm(self.sat_data['coords'][start:end] - self.sat_data['COPs'][sat_idx], axis=1)
            mass = self.sat_data['masses'][start:end]

        mask = np.argsort(r)
        r = r[mask]

        cmass = np.cumsum(mass[mask])

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
        plt.title('Satellite rotation profiles of halo with GN = %i (%s)'%(self.gn,self.dataset))
        plt.ylabel('Velocity [km/s]'); plt.xlabel('r [kpc]')
        plt.xlim(0,1.5); plt.ylim(5,35)

        #fig.savefig('../Figures/%s/RotationCurve_g%i-sg%i.png'%(self.dataset,gn,sgn))
        plt.show()
        plt.close()

SatRotationCurves(1,dataset='V1_LR_fix_127_z000p000', nfiles_part=16, nfiles_group=96)

