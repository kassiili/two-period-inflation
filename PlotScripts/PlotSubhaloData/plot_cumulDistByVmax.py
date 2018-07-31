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

    def __init__(self, dataset):

        self.dataset = dataset
        self.reader = read_data.read_data(dataset=self.dataset.dir, nfiles_part=self.dataset.nfiles_part, nfiles_group=self.dataset.nfiles_group)

        self.read_galaxies()

    def read_galaxies(self):

        vmax = self.reader.read_subhaloData('Vmax') / 100000 # cm/s to km/s
        SM = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
        SGNs = self.reader.read_subhaloData('SubGroupNumber')
        GNs = self.reader.read_subhaloData('GroupNumber')

        mask_MW_or_M31 = np.logical_or(GNs == 1, GNs == 2)
        self.vmaxSat = vmax[np.logical_and.reduce((vmax > 0, SGNs != 0, mask_MW_or_M31))]
        SMSat = SM[np.logical_and.reduce((vmax > 0, SGNs != 0, mask_MW_or_M31))]
        self.vmaxLumSat = self.vmaxSat[SMSat > 0]
        self.vmaxDarkSat = self.vmaxSat[SMSat == 0]

        self.vmaxIsol = vmax[np.logical_and(vmax > 0, SGNs == 0)]
        SMIsol = SM[np.logical_and(vmax > 0, SGNs == 0)]
        self.vmaxLumIsol = self.vmaxIsol[SMIsol > 0]
        self.vmaxDarkIsol = self.vmaxIsol[SMIsol == 0]

        #Sort arrays in descending order:
        self.vmaxSat[::-1].sort()
        self.vmaxLumSat[::-1].sort()
        self.vmaxDarkSat[::-1].sort()
        self.vmaxIsol[::-1].sort()
        self.vmaxLumIsol[::-1].sort()
        self.vmaxDarkIsol[::-1].sort()


class plot_subhalo_dist_vs_vmax:

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
        self.axes.set_xlim(7, 100)
        self.axes.set_ylim(1, 400)
        
    def set_labels(self):
        """ Set labels. """

        self.axes.set_xlabel('$v_{\mathrm{max}} [\mathrm{km s^{-1}}]$')
        self.axes.set_ylabel('$N(>v_{\mathrm{max}})$')
#        if (self.satellites):
#            self.axes.set_title('Distribution of satellites as a function of $v_{max}$')
#        else:
#            self.axes.set_title('Distribution of isolated galaxies as a function of $v_{max}$')

    def add_data(self, data, colors):
        """ Plot data into an existing figure. Satellites is a boolean variable with value 1, if satellites are to be plotted, and 0, if instead isolated galaxies are to be plotted. """

        x = 0; y = 0
        if self.satellites:
            all = data.vmaxSat; lum = data.vmaxLumSat; dark = data.vmaxDarkSat
        else:
            all = data.vmaxIsol; lum = data.vmaxLumIsol; dark = data.vmaxDarkIsol

        self.axes.plot(lum, np.arange(1, lum.size + 1), c=colors[0], label=data.dataset.name+": kirkkaat")
        self.axes.plot(dark, np.arange(1, dark.size + 1), c=colors[1], label=data.dataset.name+": pimeät")
        self.axes.plot(all, np.arange(1, all.size + 1), linestyle=':', c=colors[1], label=data.dataset.name+": kaikki")
    
    def save_figure(self, dir):
        """ Save figure. """
        
        self.axes.legend(loc=0)
        plt.show()
        filename=""
        if self.satellites:
            filename = 'Dist-of-subhaloes_vs_Vmax_sat.png'
        else:
            filename = 'Dist-of-subhaloes_vs_Vmax_isol.png'

        path = '../Figures/%s'%dir
        # If the directory does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)
        self.fig.savefig(os.path.join(path,filename))
        plt.close()



#plot = plot_subhalo_dist_vs_vmax()
#LCDM = subhalo_dist_vs_vmax_data(dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192)
#curvaton = subhalo_dist_vs_vmax_data(dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64)
#
#plot.add_data(LCDM, 1, ['lightblue', 'blue'])
#plot.add_data(curvaton, 1, ['pink', 'red'])
#plot.save_figure() 
    
