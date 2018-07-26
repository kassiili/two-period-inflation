import os, sys
import h5py
import numpy as np
import time
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class RotationCurve:

    def __init__(self, gn, sgn, dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192):

        self.dataset = dataset
        self.nfiles_part = nfiles_part
        self.nfiles_group = nfiles_group
        self.reader = read_data.read_data(dataset=self.dataset, nfiles_part=self.nfiles_part, nfiles_group=self.nfiles_group)

        self.a, self.h, self.massTable, self.boxsize = self.reader.read_header()
        self.boxsize = self.boxsize*1000/self.h # Mpc/h -> kpc
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

        subGroupNumbers = self.reader.read_subhaloData('SubGroupNumber')
        groupNumbers = self.reader.read_subhaloData('GroupNumber')
        cop = self.reader.read_subhaloData('CentreOfPotential')

        return cop[np.logical_and(subGroupNumbers == sgn, groupNumbers == gn)] * u.cm.to(u.kpc)

    def read_galaxy(self, itype, gn, sgn):
        """ For a given galaxy (identified by its GroupNumber and SubGroupNumber),
        extract the coordinates and mass of all particles of a selected type.
        Coordinates are then wrapped around the centre to account for periodicity. """

        data = {}

        # Load data, then mask to selected GroupNumber and SubGroupNumber.
        gns  = self.reader.read_dataset(itype, 'GroupNumber')
        sgns = self.reader.read_dataset(itype, 'SubGroupNumber')

        mask = np.logical_and(gns == gn, sgns == sgn)
        if itype == 1:
            data['mass'] = self.reader.read_dataset_dm_mass()[mask] * u.g.to(u.Msun)
        else:
            data['mass'] = self.reader.read_dataset(itype, 'Masses')[mask] * u.g.to(u.Msun)
        data['coords'] = self.reader.read_dataset(itype, 'Coordinates')[mask] * u.cm.to(u.kpc)

        # Periodic wrap coordinates around centre.
        data['coords'] = np.mod(data['coords']-self.centre+0.5*self.boxsize,self.boxsize)+self.centre-0.5*self.boxsize
        return data

    def compute_rotation_curve(self, arr):
        """ Compute the rotation curve. """

        # Compute distance to centre.
        #self.centre = np.mod(self.centre + 0.5*self.boxsize,self.boxsize)-0.5*self.boxsize
        
        r = np.linalg.norm(arr['coords'] - self.centre, axis=1)
       # r = r[r>0]
        mask = np.argsort(r)
        r = r[mask]

        cmass = np.cumsum(arr['mass'][mask])

        # Begin rotation curve from the 10th particle to reduce noise at the low end of the curve.
        r = r[10:]
        cmass = cmass[10:]

        # Compute velocity.
        myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
        v = np.sqrt((myG * cmass) / r)

        # Return r in kpc and v in km/s.
        return r, v

    def calc_V1kpc(self, arr):
        """ Calculate the rotational velocity at 1kpc. """

        # Compute distance to centre.
        r = np.linalg.norm(arr['coords'] - self.centre, axis=1)

        # Mask chooses only particles within 1kpc of cop.
        mask = np.logical_and(r > 0, r < 1)

        myG = G.to(u.km**2 * u.kpc * u.Msun**-1 * u.s**-2).value
        return np.sqrt(myG * arr['mass'][mask].sum())

    def plot(self, gn, sgn):
        fig, axes = plt.subplots()

        # All parttypes together.
        combined = {}
        combined['mass'] = np.concatenate((self.gas['mass'], self.dm['mass'],
            self.stars['mass'], self.bh['mass']))
        combined['coords'] = np.vstack((self.gas['coords'], self.dm['coords'],
            self.stars['coords'], self.bh['coords']))

        r, v = self.compute_rotation_curve(combined)
        axes.plot(r, v)

        # Get Vmax and Rmax:
        subGroupNumbers = self.reader.read_subhaloData('SubGroupNumber')
        groupNumbers = self.reader.read_subhaloData('GroupNumber')
        vmax = self.reader.read_subhaloData('Vmax')[np.logical_and(subGroupNumbers == sgn, groupNumbers == gn)]/100000  # cm/s to km/s
        rmax = self.reader.read_subhaloData('VmaxRadius')[np.logical_and(subGroupNumbers == sgn, groupNumbers == gn)] * u.cm.to(u.kpc)

        v1kpc = self.calc_V1kpc(combined)

        axes.axhline(vmax, linestyle='dashed', c='red', label='Vmax=%1.3f'%vmax)
        axes.axvline(rmax, linestyle='dashed', c='red', label='Rmax=%1.3f'%rmax)
        axes.axhline(v1kpc, linestyle='dashed', c='green', label='V1kpc=%1.3f'%v1kpc)
        axes.axvline(1, linestyle='dashed', c='green')
        
        # Save plot.
        axes.legend(loc=0)
        axes.minorticks_on()
        axes.set_title('Rotation curve of halo with GN = %i and SGN = %i\n(%s)'%(gn,sgn,self.dataset))
        axes.set_ylabel('Velocity [km/s]'); axes.set_xlabel('r [kpc]')
        axes.set_xlim(0, 80); # axes.tight_layout()

        plt.show()
        fig.savefig('../Figures/%s/RotationCurve_g%i-sg%i.png'%(self.dataset,gn,sgn))
        plt.close()

RotationCurve(1, 0, dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64) 


#oddg = [5, 17, 51, 80, 80]
#oddsg = [11, 2, 1, 1, 2]
#
#for g, sg in zip(oddg, oddsg):
#    RotationCurve(g,sg)
#
#oddgIsol = [121, 258, 270, 299, 519]    # 5 of many 
#
#for gn in oddgIsol:
#    RotationCurve(gn, 0)
