import os, sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
from calc_median import calc_median_trend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data

class SM_vs_Vmax_data:

    def __init__(self, dataset):

        self.dataset = dataset
        self.reader = read_data.read_data(dataset=self.dataset.dir, nfiles_part=self.dataset.nfiles_part, nfiles_group=self.dataset.nfiles_group)

        self.read_galaxies()

    def read_galaxies(self):

        # Read subhalo data:
        vmax = self.reader.read_subhaloData('Vmax') / 100000   # cm/s to km/s
        SM = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
        SGNs = self.reader.read_subhaloData('SubGroupNumber')

        # Divide into satellites and isolated galaxies:
        maskSat = np.logical_and.reduce((vmax > 0, SM > 0, SGNs != 0))
        maskIsol = np.logical_and.reduce((vmax > 0, SM > 0, SGNs == 0))

        self.vmaxSat = vmax[maskSat]
        self.SMSat = SM[maskSat]
        self.vmaxIsol = vmax[maskIsol]
        self.SMIsol = SM[maskIsol]


class plot_SM_vs_Vmax:

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
        self.axes.set_xlim(10, 100)
        self.axes.set_ylim(10**5, 5*10**9)
        
    def set_labels(self):
        """ Set labels. """

        self.axes.set_xlabel('$v_{max}[\mathrm{km s^{-1}}]$')
        self.axes.set_ylabel('$M_*[\mathrm{M_\odot}]$')
        if (self.satellites):
#            self.axes.set_title('Stellar mass of satellites')
            self.axes.text(11, 2*10**9, 'satelliittigalaksit')
        else:
#            self.axes.set_title('Stellar mass of isolated galaxies')
            self.axes.text(11, 2*10**9, 'eristetyt galaksit')

    def add_data(self, data, plot_style):
        """ Plot data (object of type SM_vs_Vmax_data) into an existing figure. Satellites is a boolean variable with value 1, if satellites are to be plotted, and 0, if instead isolated galaxies are to be plotted. """

        x = 0; y = 0
        if self.satellites:
            x = data.vmaxSat; y = data.SMSat
        else:
            x = data.vmaxIsol; y = data.SMIsol

        # Plot data points:
        self.axes.scatter(x, y, marker=plot_style[0], c=plot_style[1], edgecolor='none', label=data.dataset.name)

        if not self.satellites:
            # Plot satellite median curves:
            xSat = data.vmaxSat; ySat = data.SMSat
            median = calc_median_trend(xSat, ySat, points_per_bar=7)
            self.axes.plot(median[0], median[1], c='grey', linestyle='--')

        # Plot median:
        median = calc_median_trend(x, y, points_per_bar=7)
        self.axes.plot(median[0], median[1], c=plot_style[2], linestyle='--')
        #self.axes.scatter(median[0], median[1], s=5)
    
    def save_figure(self, dir):
        """ Save figure. """

        filename=""
        if self.satellites:
            filename = 'SM_vs_Vmax_sat.png'
            self.axes.legend(loc=0)
        else:
            filename = 'SM_vs_Vmax_isol.png'
        plt.show()

        path = '../Figures/%s'%dir
        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(os.path.join(path,filename))
        plt.close()

#
#plot = plot_SM_vs_Vmax()
#LCDM = SM_vs_Vmax_data(dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192)
#curvaton = SM_vs_Vmax_data(dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64)
#
#plot.add_data(LCDM, 1, 'red')
#plot.add_data(curvaton, 1, 'blue')
#plot.save_figure() 
    

