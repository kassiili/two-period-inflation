import os, sys
import numpy as np
import h5py
import astropy.units as u
import matplotlib.pyplot as plt
from calc_median import calc_median_trend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class plot_Rmax_vs_Vmax:

    def __init__(self, dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192):

        self.dataset = dataset
        self.nfiles_part = nfiles_part
        self.nfiles_group = nfiles_group
        self.reader = read_data.read_data(dataset=self.dataset, nfiles_part=self.nfiles_part, nfiles_group=self.nfiles_group)

        maxVelocities = self.reader.read_subhaloData('Vmax') / 100000   # cm/s to km/s
        maxRadii = self.reader.read_subhaloData('VmaxRadius') * u.cm.to(u.kpc)
        self.stellarMasses = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
        self.subGroupNumbers = self.reader.read_subhaloData('SubGroupNumber')

        maskSat = np.logical_and.reduce((maxVelocities > 0, maxRadii > 0, self.subGroupNumbers != 0, self.stellarMasses > 0))
        maskIsol = np.logical_and.reduce((maxVelocities > 0, maxRadii > 0, self.subGroupNumbers == 0, self.stellarMasses > 0))

        self.maxVelocitiesSat = maxVelocities[maskSat]
        self.maxRadiiSat = maxRadii[maskSat]
        self.maxVelocitiesIsol = maxVelocities[maskIsol]
        self.maxRadiiIsol = maxRadii[maskIsol]

    def plot(self):
        fig,axes = plt.subplots()

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.scatter(self.maxVelocitiesSat, self.maxRadiiSat, s=3, c='red', edgecolor='none', label='satelliittigalaksit')
        axes.scatter(self.maxVelocitiesIsol, self.maxRadiiIsol, s=3, c='blue', edgecolor='none', label='eristetyt galaksit')

        median = calc_median_trend(self.maxVelocitiesSat, self.maxRadiiSat)
        axes.plot(median[0], median[1], c='red', linestyle='--')

        median = calc_median_trend(self.maxVelocitiesIsol, self.maxRadiiIsol)
        axes.plot(median[0], median[1], c='blue', linestyle='--')

        plt.xlim(20, 150);
        plt.ylim(1, 50);

        axes.legend(loc=0)
        axes.set_xlabel('$v_{\mathrm{max}}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$r_{\mathrm{max}}[\mathrm{kpc}]$')
        axes.set_title('Max circular velocities and corresponding radii\n(%s)'%self.dataset)

        plt.show()
        fig.savefig('../Figures/%s/rmax_vs_vmax.png'%self.dataset)
        plt.close()

plot = plot_Rmax_vs_Vmax() #dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64) 
plot.plot()

