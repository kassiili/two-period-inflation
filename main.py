import matplotlib.pyplot as plt
import numpy as np
import h5py

from dataset import Dataset
from PlotSubhaloData.plot_SM_vs_Vmax import plot_SM_vs_Vmax

# Get data:
LCDM = Dataset("V1_LR_fix_127_z000p000", "LCDM")
vmaxSat, vmaxIsol = LCDM.get_subhalos("Vmax",True)
SMSat, SMIsol = LCDM.get_subhalos("Stars/Mass",True)

# Exclude dark halos:
maskSat = np.logical_and.reduce((vmaxSat>0, SMSat>0))
maskIsol = np.logical_and.reduce((vmaxIsol>0, SMIsol>0))
vmaxSat = vmaxSat[maskSat]; SMSat = SMSat[maskSat]
vmaxIsol = vmaxIsol[maskIsol]; SMIsol = SMIsol[maskIsol]
print(np.amax(vmaxSat))
print(np.amin(vmaxSat))

fig, axes = plt.subplots()
plot = plot_SM_vs_Vmax(axes, True)
plot.add_scatter((SMSat,vmaxSat), 'pink', LCDM.name)
plt.show()

