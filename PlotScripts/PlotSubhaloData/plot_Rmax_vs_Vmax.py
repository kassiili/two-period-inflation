import os, sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from calc_median import calc_median_trend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data

class rmax_vs_vmax_data:

    def __init__(self, dataset):

        self.dataset = dataset
        self.reader = read_data.read_data(dataset=self.dataset.dir, nfiles_part=self.dataset.nfiles_part, nfiles_group=self.dataset.nfiles_group)

        self.read_galaxies()

    def read_galaxies(self):

        vmax = self.reader.read_subhaloData('Vmax') / 100000   # cm/s to km/s
        rmax = self.reader.read_subhaloData('VmaxRadius') * u.cm.to(u.kpc)
        self.SM = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
        self.SGNs = self.reader.read_subhaloData('SubGroupNumber')

        maskSat = np.logical_and.reduce((vmax > 0, rmax > 0, self.SGNs != 0, self.SM > 0))
        maskIsol = np.logical_and.reduce((vmax > 0, rmax > 0, self.SGNs == 0, self.SM > 0))

        self.vmaxSat = vmax[maskSat]
        self.rmaxSat = rmax[maskSat]
        self.vmaxIsol = vmax[maskIsol]
        self.rmaxIsol = rmax[maskIsol]


class plot_rmax_vs_vmax:

    def __init__(self, satellites):
        """ Create new figure with stellar mass on y-axis and Vmax on x-axis. """
    
        self.fig, self.axes = plt.subplots()
        self.satellites = satellites
        self.set_axes()
        self.set_labels()
        
    def set_axes(self):
        """ Set shapes for axes. """

        self.axes.set_xscale('log')
        self.axes.set_yscale('log')

        self.axes.set_xlim(20, 150);
        self.axes.set_ylim(1, 50);
        
    def set_labels(self):
        """ Set labels. """

        self.axes.set_xlabel('$v_{\mathrm{max}}[\mathrm{km s^{-1}}]$')
        self.axes.set_ylabel('$r_{\mathrm{max}}[\mathrm{kpc}]$')
        if self.satellites:
            self.axes.set_title('Satellite max circular velocities and corresponding radii')
        else:
            self.axes.set_title('Isolated galaxy max circular velocities and corresponding radii')

    def add_data(self, data, col):
        """ Plot data into an existing figure. Satellites is a boolean variable with value 1, if satellites are to be plotted, and 0, if instead isolated galaxies are to be plotted. """

        x = 0; y = 0
        if self.satellites:
            x = data.vmaxSat; y = data.rmaxSat
            median = calc_median_trend(data.vmaxSat, data.rmaxSat)
        else:
            x = data.vmaxIsol; y = data.rmaxIsol
            median = calc_median_trend(data.vmaxIsol, data.rmaxIsol)

        self.axes.scatter(x, y, s=3, c=col, edgecolor='none', label=data.dataset.name)
        self.axes.plot(median[0], median[1], c=col, linestyle='--')
    
    def save_figure(self, dir):
        """ Save figure. """
        
        self.axes.legend(loc=0)
        plt.show()
        filename=""
        if self.satellites:
            filename = 'rmax_vs_vmax_sat.png'
        else:
            filename = 'rmax_vs_vmax_isol.png'

        path = '../Figures/%s'%dir
        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(os.path.join(path,filename))
        plt.close()
