import sys, os
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data


class satellites_count:

    def __init__(self, radius, dataset):

        self.dataset = dataset
        self.radius = radius
        self.reader = read_data.read_data(dataset=self.dataset.dir, nfiles_part=self.dataset.nfiles_part, nfiles_group=self.dataset.nfiles_group)

        self.calc_satellites()

    def calc_satellites(self):
    
        sgns = self.reader.read_subhaloData('SubGroupNumber')
        gns = self.reader.read_subhaloData('GroupNumber')
        COPs = self.reader.read_subhaloData('CentreOfPotential') * u.cm.to(u.kpc)
        masses = self.reader.read_subhaloData('Mass') * u.g.to(u.Msun) / 10**12
        SM = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
    
        # Get LG mass:
        MW_mask = np.logical_and(gns == 1, sgns == 0)
        M31_mask = np.logical_and(gns == 2, sgns == 0)
        self.mass = (masses[MW_mask] + masses[M31_mask])

        # Calculate barycentre and mass of the LG-analogue:
        centre = (masses[MW_mask] * COPs[MW_mask] + masses[M31_mask] * COPs[M31_mask]) / self.mass
    
        # Calculate distances to the barycentre:
        r = np.linalg.norm(COPs - centre, axis=1)
    
        # Calculate the number of satellites within the radius from the barycentre:
        self.sat_cnt = np.sum(np.logical_and.reduce((sgns != 0, r < self.radius, SM > 10**5)))


class plot_satellites_count:

    def __init__(self):
        """ Create new figure with stellar mass on y-axis and Vmax on x-axis. """
    
        self.fig, self.axes = plt.subplots()
        self.set_labels()
        
    def set_labels(self):
        """ Set labels. """

        plt.xlabel('$M_{200}[\mathrm{10^{12} M_\odot}]$')
        plt.ylabel('$N(r<300\mathrm{ kpc})$')

    def add_data(self, data, col):
        """ Plot data into an existing figure. Satellites is a boolean variable with value 1, if satellites are to be plotted, and 0, if instead isolated galaxies are to be plotted. """

        self.radius = data.radius
        plt.scatter(data.mass, data.sat_cnt, c=col, label=data.dataset.name)
    
    def save_figure(self, dir):
        """ Save figure. """
        
        self.axes.legend(loc=0)
        plt.show()
        filename="satellites_in_LG_r%i.py"%self.radius

        path = '../Figures/%s'%dir
        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(os.path.join(path,filename))
        plt.close()

