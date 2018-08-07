import os,sys
import numpy as np
import astropy.units as u
import random
from dataset import dataset
from plot_rotation_curve import plot_rotation_curve, rotation_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ReadData"))
import read_data 

def read_random_satellites(dataset):

    reader = read_data.read_data(dataset.dir, dataset.nfiles_part, dataset.nfiles_group)
    vmax = reader.read_subhaloData('Vmax') * u.cm.to(u.km)
    SM = reader.read_subhaloData('Stars/Mass') * u.g.to(u.Msun)
    gns = reader.read_subhaloData('GroupNumber')
    sgns = reader.read_subhaloData('SubGroupNumber')
    
    cops = reader.read_subhaloData('CentreOfPotential') * u.cm.to(u.kpc)
    cop_M31 = cops[np.logical_and(gns==1,sgns==0)]
    cop_MW = cops[np.logical_and(gns==2,sgns==0)]
    d_M31 = np.linalg.norm(cops - cop_M31, axis=1)
    d_MW = np.linalg.norm(cops - cop_MW, axis=1)
    
    # Choose luminous satellites with vmax=size+-range:
    size = 20; range = 2
    mask = np.logical_and.reduce((vmax > size-range, 
        vmax < size+range, 
        SM > 0, 
        np.logical_or(d_M31 < 300, d_MW < 300)))
    gns = gns[mask]; sgns = sgns[mask]
    idxs = np.random.choice(gns.size, 7, replace=False)
    print(gns.size)

    return gns.take(idxs), sgns.take(idxs)

LCDM_dataset = dataset('V1_MR_fix_082_z001p941', 'standard', 16, 192)
mock_dataset = dataset('V1_MR_mock_1_fix_082_z001p941', 'curvaton', 1, 64)

plot = plot_rotation_curve(-1, -1)
gns, sgns = read_random_satellites(LCDM_dataset)
label = 1
for gn,sgn in zip(gns, sgns):
    print(gn, ', ', sgn)
    plot.add_data(rotation_curve(gn, sgn, LCDM_dataset), 'red', 0, label)
    label=0

gns, sgns = read_random_satellites(mock_dataset)
label = 1
for gn,sgn in zip(gns, sgns):
    print(gn, ', ', sgn)
    plot.add_data(rotation_curve(gn, sgn, mock_dataset), 'blue', 0, label)
    label=0

plot.axes.axvline(1, linestyle='dashed', c='black')

plot.save_figure('RotationCurves/Comparisons_082_z001p941') 
