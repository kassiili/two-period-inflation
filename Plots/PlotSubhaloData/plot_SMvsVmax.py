import sys
import numpy as np
import h5py
import time
import astropy.units as u
import matplotlib.pyplot as plt
from read_subhaloData import read_subhaloData

sys.path.insert(0, '/home/kassiili/SummerProject/practise-with-datasets/Plots/')
from read_header import read_header

class plot_SM_vs_Vmax:

    def __init__(self, dataset='LR'):
        self.dataset = dataset
        maxVelocities = read_subhaloData('Vmax', dataset=self.dataset) / 100000   # cm/s to km/s
        stellarMasses = read_subhaloData('Stars/Mass', dataset=self.dataset) * u.g.to(u.Msun)
        self.subGroupNumbers = read_subhaloData('SubGroupNumber', dataset=self.dataset)
        maskSat = np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, self.subGroupNumbers != 0))
        maskIsol = np.logical_and.reduce((maxVelocities > 0, stellarMasses > 0, self.subGroupNumbers == 0))

        self.maxVelocitiesSat = maxVelocities[maskSat]
        self.stellarMassesSat = stellarMasses[maskSat]
        self.maxVelocitiesIsol = maxVelocities[maskIsol]
        self.stellarMassesIsol = stellarMasses[maskIsol]

    def plot(self):
        fig = plt.figure()
        axes = plt.gca()

        axes.set_xscale('log')
        axes.set_yscale('log')

        axes.scatter(self.maxVelocitiesSat, self.stellarMassesSat, s=3, c='red', edgecolor='none', label='satellite galaxies')
        axes.scatter(self.maxVelocitiesIsol, self.stellarMassesIsol, s=3, c='blue', edgecolor='none', label='isolated galaxies')

        median = self.calc_median_trend2(self.maxVelocitiesSat, self.stellarMassesSat)
        axes.plot(median[0], median[1], c='red', linestyle='--')

        median = self.calc_median_trend2(self.maxVelocitiesIsol, self.stellarMassesIsol)
        axes.plot(median[0], median[1], c='blue', linestyle='--')

        axes.legend()
        axes.set_xlabel('$v_{max}[\mathrm{km s^{-1}}]$')
        axes.set_ylabel('$M_*[\mathrm{M_\odot}]$')
        #axes.set_title('Stellar mass of luminous subhaloes')

        plt.show()
#        plt.savefig('SM_vs_Vmax_%s.png'%self.dataset)
#        plt.close()

    def calc_median_trend(self, x, y):
        minX = min(x)
        maxX = max(x)
        bars = 10
        tmpX = np.linspace(minX, maxX, num=bars)
        medianTrend = np.empty([bars-1,])
        for i in range(bars-1):
            print(y[np.logical_and(tmpX[i] < x, x < tmpX[i+1])], '\n')
            medianTrend[i] = np.median(y[np.logical_and(tmpX[i] < x, x < tmpX[i+1])])

        return [tmpX[:-1] + (maxX-minX)/(2*bars), medianTrend] 

    def calc_median_trend2(self, x, y):
        xy = np.vstack([x,y])
        xy = xy[:,xy[0,:].argsort()]
        bars = 10
        datapoints = xy[0,:].size

        #Excĺude first elements (if necessary) to allow reshaping:
        if (datapoints%bars != 0):
            tmpX = xy[0,(datapoints%bars):]
            tmpY = xy[1,(datapoints%bars):]
            datapoints -= datapoints%bars
        else:
            tmpX = xy[0,:]
            tmpY = xy[1,:]

        #Reshape into a 2D numpy array with number of rows = bars:
        tmpX = tmpX.reshape((bars, int(datapoints/bars)))
        tmpY = tmpY.reshape((bars, int(datapoints/bars)))

        #Calculate the medians of the rows of the reshaped arrays:
        medianX = np.median(tmpX, axis=1)
        medianY = np.median(tmpY, axis=1)

        return [medianX, medianY]

plot = plot_SM_vs_Vmax(dataset='MR') 
plot.plot()

