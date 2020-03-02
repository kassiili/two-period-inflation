import os, sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
from PlotSubhaloData.calc_median import calc_median_trend

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

    def __init__(self, ax, satellites):
        """ Create new figure with stellar mass on y-axis and Vmax on x-axis. """
    
        self.ax = ax
        #self.set_axes()
        self.set_labels(satellites)
        self.n_datasets = 0
        
    def set_axes(self):
        """ Set shapes for axes. """

        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlim(10, 100)
        self.ax.set_ylim(10**5, 5*10**9)
        
    def set_labels(self, satellites):
        """ Set labels. """

        self.ax.set_xlabel('$v_{max}[\mathrm{km s^{-1}}]$', fontsize=16)
        self.ax.set_ylabel('$M_*[\mathrm{M_\odot}]$', fontsize=16)
        if (satellites):
#            self.axes.set_title('Stellar mass of satellites')
            self.ax.text(11, 2*10**9, 'satelliittigalaksit')
        else:
#            self.axes.set_title('Stellar mass of isolated galaxies')
            self.ax.text(11, 2*10**9, 'eristetyt galaksit')

    def add_scatter(self, data, color, label):
        """ Plot data (object of type SM_vs_Vmax_data) into an existing figure. Satellites is a boolean variable with value 1, if satellites are to be plotted, and 0, if instead isolated galaxies are to be plotted. """

        marker = (3, self.n_datasets % 4, 0)

        self.ax.scatter(data[0], data[1], marker=marker, c=color, \
                edgecolor='none', label=label)

        self.n_datasets += 1

    def add_median(self, data, color):

        median = calc_median_trend(x, y, points_per_bar=7)
        self.ax.plot(median[0], median[1], c=color, linestyle='--')
    
#    def get_filename(self, dir):
#        """ Save figure. """
#
#        filename=""
#        if self.satellites:
#            filename = 'SM_vs_Vmax_sat.png'
#            self.ax.legend(loc=0)
#        else:
#            filename = 'SM_vs_Vmax_isol.png'
#
#        path = '../Figures/%s'%dir
#        # If the directory does not exist, create it
#        if not os.path.exists(path):
#            os.makedirs(path)
#        self.fig.savefig(os.path.join(path,filename))

#
#plot = plot_SM_vs_Vmax()
#LCDM = SM_vs_Vmax_data(dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192)
#curvaton = SM_vs_Vmax_data(dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64)
#
#plot.add_data(LCDM, 1, 'red')
#plot.add_data(curvaton, 1, 'blue')
#plot.save_figure() 
    

