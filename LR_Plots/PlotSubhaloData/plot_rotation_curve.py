import sys
import h5py
import numpy as np
import time
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
        self.centre = self.find_centre_of_potential(gn, sgn)

        # Load data.
        self.gas    = self.read_galaxy(0, gn, sgn)
        self.dm     = self.read_galaxy(1, gn, sgn)
        self.stars  = self.read_galaxy(4, gn, sgn)
        self.bh     = self.read_galaxy(5, gn, sgn)

        # Plot.
        self.plot(gn, sgn)

    def find_centre_of_potential(self, gn, sgn):
        """ Obtain the centre of potential of the given galaxy from the subhalo catalogues. """

        subGroupNumbers = read_subhaloData('SubGroupNumber')
        groupNumbers = read_subhaloData('GroupNumber')
        cop = read_subhaloData('CentreOfPotential')

        return cop[np.logical_and(subGroupNumbers == sgn, groupNumbers == gn)] * u.cm.to(u.Mpc)

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

        # Begin rotation curve from the 10th particle to reduce noise at the low end of the curve.
        r = r[range(9, r.size)]
        cmass = cmass[range(9, cmass.size)]

        # Compute velocity.
        myG = G.to(u.km**2 * u.Mpc * u.Msun**-1 * u.s**-2).value
        v = np.sqrt((myG * cmass) / r)

        # Return r in Mpc and v in km/s.
        return r, v

    def calc_V1kpc(self, arr):
        """ Calculate the rotational velocity at 1kpc. """

        # Compute distance to centre.
        r = np.linalg.norm(arr['coords'] - self.centre, axis=1)

        # Mask chooses only particles within 1kpc of cop.
        mask = np.logical_and(r > 0, r < 10**-3)

        myG = G.to(u.km**2 * u.Mpc * u.Msun**-1 * u.s**-2).value
        return np.sqrt(myG * arr['mass'][mask].sum() / 10**-3)


    def plot(self, gn, sgn):
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
        plt.plot(r*1000., v)

        # Get Vmax and Rmax:
        subGroupNumbers = read_subhaloData('SubGroupNumber')
        groupNumbers = read_subhaloData('GroupNumber')
        vmax = read_subhaloData('Vmax')[np.logical_and(subGroupNumbers == sgn, groupNumbers == gn)]/100000  # cm/s to km/s
        rmax = read_subhaloData('VmaxRadius')[np.logical_and(subGroupNumbers == sgn, groupNumbers == gn)] * u.cm.to(u.kpc)

        v1kpc = self.calc_V1kpc(combined)

        plt.axhline(vmax, linestyle='dashed', c='red', label='Vmax=%1.3f'%vmax)
        plt.axvline(rmax, linestyle='dashed', c='red', label='Rmax=%1.3f'%rmax)
        plt.axhline(v1kpc, linestyle='dashed', c='green', label='V1kpc=%1.3f'%v1kpc)
        plt.axvline(1, linestyle='dashed', c='green')
        
        # Save plot.
        plt.legend(loc='center right')
        plt.minorticks_on()
        plt.title('Rotation curve of halo with GN = %i and SGN = %i'%(gn,sgn))
        plt.ylabel('Velocity [km/s]'); plt.xlabel('r [kpc]')
        plt.xlim(0, 50); # plt.tight_layout()

#        plt.show()
        plt.savefig('RotationCurve_g%i-sg%i.png'%(gn,sgn))
        plt.close()

#oddsg = [20, 25, 27, 1]
#oddg = [2, 3, 3, 51]

RotationCurve(1,0)
#RotationCurve(2,5)
#RotationCurve(1,16)
#RotationCurve(9,3)

#for g, sg in zip(oddg, oddsg):
#    print(g, sg)
#    RotationCurve(g,sg)

#gn = 4
#sgns = [0,1,2]
#for sgn in sgns:
#    x = RotationCurve(gn, sgn)

