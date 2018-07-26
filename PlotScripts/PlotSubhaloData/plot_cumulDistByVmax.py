import os, sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data as read_data

class plot_subhalo_dist_vs_vmax:

    def __init__(self, dataset='V1_LR_fix_127_z000p000', nfiles_part=16, nfiles_group=96):

        self.dataset = dataset
        self.nfiles_part = nfiles_part
        self.nfiles_group = nfiles_group
        self.reader = read_data.read_data(dataset=self.dataset, nfiles_part=self.nfiles_part, nfiles_group=self.nfiles_group)

        maxVelocities = self.reader.read_subhaloData('Vmax') / 100000 # cm/s to km/s
        stellarMasses = self.reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
        subGroupNumbers = self.reader.read_subhaloData('SubGroupNumber')
        self.maxVelocitiesSatLum = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, subGroupNumbers != 0))]
        self.maxVelocitiesSatDark = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses == 0, subGroupNumbers != 0))]
        self.maxVelocitiesIsolLum = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, subGroupNumbers == 0))]
        self.maxVelocitiesIsolDark = maxVelocities[np.logical_and.reduce((maxVelocities > 0, stellarMasses == 0, subGroupNumbers == 0))]

        #Sort arrays in descending order:
        self.maxVelocitiesSatLum[::-1].sort()
        self.maxVelocitiesSatDark[::-1].sort()
        self.maxVelocitiesIsolLum[::-1].sort()
        self.maxVelocitiesIsolDark[::-1].sort()

    def plot(self):
        fig,axes = plt.subplots()

        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.plot(self.maxVelocitiesSatLum, np.arange(1, self.maxVelocitiesSatLum.size + 1), c='lightblue', label='Kirkkaat satelliittigalaksit')
        axes.plot(self.maxVelocitiesSatDark, np.arange(1, self.maxVelocitiesSatDark.size + 1), c='blue', label='Pimeät satelliittigalaksit')
        axes.plot(self.maxVelocitiesIsolLum, np.arange(1, self.maxVelocitiesIsolLum.size + 1), c='pink', label='Kirkkaat eristetyt galaksit')
        axes.plot(self.maxVelocitiesIsolDark, np.arange(1, self.maxVelocitiesIsolDark.size + 1), c='red', label='Pimeät eristetyt galaksit')

        axes.legend(loc=0)
        axes.set_xlabel('$v_{\mathrm{max}} [\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$N(>v_{\mathrm{max}})$')
        axes.set_title('Distribution of luminous subhaloes as a function of $v_{max}$\n(%s)'%self.dataset)

        #plt.show()
        fig.savefig('../Figures/%s/Dist-of-subhaloes_vs_Vmax.png'%self.dataset)
        plt.close()

plot = plot_subhalo_dist_vs_vmax(dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192)
plot.plot()

