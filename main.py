import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy import units

from dataset import Snapshot
from PlotSubhaloData.plot_SM_vs_Vmax import plot_SM_vs_Vmax

# Get data:
LCDM = Snapshot("V1_LR_fix_127_z000p000", "LCDM")
vmaxSat, vmaxIsol = LCDM.get_subhalos("Vmax",True)
SMSat, SMIsol = LCDM.get_subhalos("Stars/Mass",True)

# Exclude dark halos
maskSat = np.logical_and.reduce((vmaxSat>0, SMSat>0))
maskIsol = np.logical_and.reduce((vmaxIsol>0, SMIsol>0))

# Convert to proper units:
vmaxSat = vmaxSat[maskSat] / 100000 # cm/s to km/s 
vmaxIsol = vmaxIsol[maskIsol] / 100000
SMSat = SMSat[maskSat] * units.g.to(units.Msun)
SMIsol = SMIsol[maskIsol] * units.g.to(units.Msun)
print(np.amax(vmaxSat))
print(np.amin(vmaxSat))

fig, axes = plt.subplots()
plot = plot_SM_vs_Vmax(axes, True)
plot.add_scatter((vmaxSat,SMSat), 'pink', LCDM.name)
plt.show()

