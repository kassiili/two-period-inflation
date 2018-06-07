import sys
import h5py
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 
from read_subhaloData_cgs_LR import read_subhaloData

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/LR_Plots/PlotPartPos/')
from read_dataset_cgs import read_dataset
from read_dataset_dm_mass import read_dataset_dm_mass
from read_header import read_header

class RotationCurve:

    def __init__(self, gn, sgn):

        self.a, self.h, self.massTable, self.boxsize = read_header()
        self.centre = self.find_center_of_mass(gn, sgn)

        # Load data.
        self.gas    = self.read_galaxy(0, gn, sgn)
        print(1)
        self.dm     = self.read_galaxy(1, gn, sgn)
        print(2)
        self.stars  = self.read_galaxy(4, gn, sgn)
        print(3)
        self.bh     = self.read_galaxy(5, gn, sgn)
        print(4)

        # Plot.
        self.plot()

    def find_center_of_mass(self, gn, sgn):
        """ Obtain the center of mass of the given galaxy from the subhalo catalogues. """

        subGroupNumbers = read_subhaloData('SubGroupNumber')
        groupNumbers = read_subhaloData('GroupNumber')
        com = read_subhaloData('CentreOfMass')

        return com[np.logical_and(subGroupNumbers == sgn, groupNumbers == gn)]

    def read_galaxy(self, itype, gn, sgn):
        """ For a given galaxy (defined by its GroupNumber and SubGroupNumber)
        extract the coordinates and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity. """

        data = {}

        # Load data, then mask to selected GroupNumber and SubGroupNumber.
        gns  = read_dataset(itype, 'GroupNumber')
        sgns = read_dataset(itype, 'SubGroupNumber')
        mask = np.logical_and(gns == gn, sgns == sgn)
        if itype == 1:
            data['mass'] = read_dataset_dm_mass()[mask] * u.g.to(u.Msun)
        else:
            data['mass'] = read_dataset(itype, 'Masses')[mask] * u.g.to(u.Msun)
        data['coords'] = read_dataset(itype, 'Coordinates')[mask] * u.cm.to(u.Mpc)

        # Periodic wrap coordinates around centre.
        boxsize = self.boxsize/self.h
        data['coords'] = np.mod(data['coords']-self.centre+0.5*boxsize,boxsize)+self.centre-0.5*boxsize

        return data

    def compute_rotation_curve(self, arr):
        """ Compute the rotation curve. """

        # Compute distance to centre.
        r = np.linalg.norm(arr['coords'] - self.centre, axis=1)
        r = r[r>0]
        mask = np.argsort(r)
        r = r[mask]

        # Compute cumulative mass.
        cmass = np.cumsum(arr['mass'][mask])

        # Compute velocity.
        myG = G.to(u.km**2 * u.Mpc * u.Msun**-1 * u.s**-2).value
        v = np.sqrt((myG * cmass) / r)

        # Return r in Mpc and v in km/s.
        return r, v

    def plot(self):
        plt.figure()

        # All parttypes together.
        combined = {}
        combined['mass'] = np.concatenate((self.gas['mass'], self.dm['mass'],
            self.stars['mass'], self.bh['mass']))
        combined['coords'] = np.vstack((self.gas['coords'], self.dm['coords'],
            self.stars['coords'], self.bh['coords']))
        
        # Loop over each parttype.
#        for x, lab in zip([self.gas, self.dm, self.stars, combined],
#                        ['Gas', 'Dark Matter', 'Stars', 'All']):
#            r, v = self.compute_rotation_curve(x)
#            plt.plot(r*1000., v, label=lab)

        r, v = self.compute_rotation_curve(combined)
        print(r.size)
        plt.plot(r*1000., v)

        # Save plot.
        #plt.legend(loc='center right')
        plt.minorticks_on()
        plt.ylabel('Velocity [km/s]'); plt.xlabel('r [kpc]')
        plt.xlim(1, 50); # plt.tight_layout()

        plt.show()
#        plt.savefig('RotationCurve.png')
#        plt.close()

x = RotationCurve(1, 0)

