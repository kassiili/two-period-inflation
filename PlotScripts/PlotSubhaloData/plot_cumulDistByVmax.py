import os, sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data

class subhalo_dist_vs_vmax_data:

    def __init__(self, dataset, gn):

        self.dataset = dataset
        self.gn = gn
        self.reader = read_data.read_data(dataset=self.dataset.dir, nfiles_part=self.dataset.nfiles_part, nfiles_group=self.dataset.nfiles_group)

        self.read_galaxies()

    def read_galaxies(self):

        vmax = self.reader.read_subhaloData('Vmax') / 100000 # cm/s to km/s
        SM = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
        SGNs = self.reader.read_subhaloData('SubGroupNumber')
        GNs = self.reader.read_subhaloData('GroupNumber')

        self.vmaxAll = vmax[np.logical_and.reduce((vmax > 0, SGNs != 0, GNs == self.gn))]
        SMAll = SM[np.logical_and.reduce((vmax > 0, SGNs != 0, GNs == self.gn))]
        self.vmaxLum = self.vmaxAll[SMAll > 0]
        self.vmaxDark = self.vmaxAll[SMAll == 0]

        #Sort arrays in descending order:
        self.vmaxAll[::-1].sort()
        self.vmaxLum[::-1].sort()
        self.vmaxDark[::-1].sort()


class plot_subhalo_dist_vs_vmax:

    def __init__(self):
        """ Create new figure with stellar mass on y-axis and Vmax on x-axis. """
    
        self.fig, self.axes = plt.subplots()
        self.data = []
        self.set_axes()
        
    def set_axes(self):
        """ Set shapes for axes. """

        self.axes.set_xscale('log')
        self.axes.set_yscale('log')
        self.axes.set_xlim(7, 100)
        self.axes.set_ylim(1, 500)
        
    def set_labels(self):
        """ Set labels. """

        self.axes.set_xlabel('$v_{\mathrm{max}} [\mathrm{km s^{-1}}]$')
        self.axes.set_ylabel('$N(>v_{\mathrm{max}})$')
        if all(item.gn == 1 for item in self.data):
            galaxy = 'M31 satelliitit'
            self.axes.text(8, 300, galaxy)
        elif all(item.gn == 2 for item in self.data):
            galaxy = 'MW satelliitit'
            self.axes.text(8, 300, galaxy)
#        self.axes.set_title('Distribution of satellites as a function of $v_{max}$')

    def add_data(self, data, colours, label):
        """ Plot data into an existing figure. Satellites is a boolean variable with value 1, if satellites are to be plotted, and 0, if instead isolated galaxies are to be plotted. """

        self.data.append(data)

        if label=='galaxy':
            if data.gn == 1:
                label_prefix = 'M31'
            else:
                label_prefix = 'MW'
        elif label=='dataset':
            label_prefix = data.dataset.name
        self.axes.plot(data.vmaxDark, np.arange(1, data.vmaxDark.size + 1), linestyle=':', c=colours[1], label=label_prefix+": pimeät")
        self.axes.plot(data.vmaxLum, np.arange(1, data.vmaxLum.size + 1), c=colours[0], label=label_prefix+": kirkkaat")
        self.axes.plot(data.vmaxAll, np.arange(1, data.vmaxAll.size + 1), c=colours[1], label=label_prefix+": kaikki")
    
    def save_figure(self, dir):
        """ Save figure. """
        
        self.set_labels()
        filename=""
        if self.data[0].gn == 1:
            filename = 'Dist-of-M31-satellites_vs_Vmax.png'
        else:
            filename = 'Dist-of-MW-satellites_vs_Vmax.png'
            self.axes.legend(loc=0)
        plt.show()

        path = '../Figures/%s'%dir
        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(os.path.join(path,filename))
        plt.close()

