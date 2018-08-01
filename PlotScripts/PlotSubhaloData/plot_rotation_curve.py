import os, sys
import h5py
import numpy as np
import time
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class rotation_curve:

    def __init__(self, gn, sgn, dataset):

        self.gn = gn; self.sgn = sgn
        self.dataset = dataset
        self.reader = read_data.read_data(dataset=self.dataset.dir, nfiles_part=self.dataset.nfiles_part, nfiles_group=self.dataset.nfiles_group)

        self.a, self.h, self.massTable, self.boxsize = self.reader.read_header()
        self.boxsize = self.boxsize*1000/self.h # Mpc/h -> kpc

        self.read_galaxy()

        self.r, self.v = self.compute_rotation_curve(self.combined)

        self.v1kpc = self.calc_V1kpc(self.combined)
        
    def read_galaxy(self):

        # Get Vmax and Rmax:
        self.SGNs = self.reader.read_subhaloData('SubGroupNumber')
        self.GNs = self.reader.read_subhaloData('GroupNumber')
        self.vmax = self.reader.read_subhaloData('Vmax')[np.logical_and(self.SGNs == self.sgn, self.GNs == self.gn)]/100000  # cm/s to km/s
        self.rmax = self.reader.read_subhaloData('VmaxRadius')[np.logical_and(self.SGNs == self.sgn, self.GNs == self.gn)] * u.cm.to(u.kpc)

        self.centre = self.find_centre_of_potential(self.gn, self.sgn) 

        # Load data.
        gas    = self.read_particles(0, self.gn, self.sgn)
        dm     = self.read_particles(1, self.gn, self.sgn)
        stars  = self.read_particles(4, self.gn, self.sgn)
        bh     = self.read_particles(5, self.gn, self.sgn)

        # All parttypes together.
        self.combined = {}
        self.combined['mass'] = np.concatenate((gas['mass'], dm['mass'], stars['mass'], bh['mass']))
        self.combined['coords'] = np.vstack((gas['coords'], dm['coords'], stars['coords'], bh['coords']))

    def find_centre_of_potential(self, gn, sgn):
        """ Obtain the centre of potential of the given galaxy from the subhalo catalogues. """

        cop = self.reader.read_subhaloData('CentreOfPotential')

        return cop[np.logical_and(self.SGNs == sgn, self.GNs == gn)] * u.cm.to(u.kpc)

    def read_particles(self, itype, gn, sgn):
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


class plot_rotation_curve:

    def __init__(self, gn, sgn):
        """ Create new figure. """
    
        self.fig, self.axes = plt.subplots()
        self.gn = gn; self.sgn = sgn
        self.set_axes()
        self.set_labels()
        
    def set_axes(self):
        """ Set shapes for axes. """

        self.axes.set_xlim(0, 80); # self.axes.tight_layout()
        self.axes.minorticks_on()

    def set_labels(self):
        """ Set labels. """

        #self.axes.set_title('Rotation curve of halo with GN = %i and SGN = %i'%(self.gn,self.sgn))
        self.axes.set_ylabel('Velocity [km/s]'); self.axes.set_xlabel('r [kpc]')

    def add_data(self, data, col, lines):
        """ Plot data into an existing figure. Satellites is a boolean variable with value 1, if satellites are to be plotted, and 0, if instead isolated galaxies are to be plotted. """

        self.axes.plot(data.r, data.v, c=col, label='%s: Vmax=%1.3f, Rmax=%1.3f, V1kpc=%1.3f'%(data.dataset.name, data.vmax, data.rmax, data.v1kpc))
        if lines:
            self.axes.axhline(data.vmax, linestyle='dashed', c=col)
            self.axes.axvline(data.rmax, linestyle='dashed', c=col)
            self.axes.axhline(data.v1kpc, linestyle='dashed', c='black')
            self.axes.axvline(1, linestyle='dashed', c='black')

    def save_figure(self,dir):
        """ Save figure. """
        
        #self.axes.legend(loc=0)
        plt.show()

        path = '../Figures/%s'%dir
        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
        #filename = 'RotationCurve_g%i-sg%i.png'%(self.gn,self.sgn)
        filename = 'RotationCurves_vmax20.png'
        self.fig.savefig(os.path.join(path,filename))
        plt.close()


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
