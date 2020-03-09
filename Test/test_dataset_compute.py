import sys,os 
import numpy as np
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dataset
from curve_fit import poly_fit
import dataset_compute

def test_calculate_V1kpc(dataset):
    v1kpc = dataset_compute.calculate_V1kpc_slow(dataset)
    sgns = dataset.get_subhalos('SubGroupNumber', divided=False)[0]
    print(v1kpc.size)
    print(sgns.size)
    print(v1kpc)
    print(sgns)
    print(np.min(v1kpc),np.max(v1kpc))

def test_get_subhalo_part_idx(dataset):
    idx = dataset_compute.get_subhalo_part_idx(dataset)
    i=0
    for l in idx:
        if sum(l) != 0:
            print(sum(l))
            i+=1
    print(i)

LCDM = Dataset("V1_LR_fix_127_z000p000","LCDM")
test_calculate_V1kpc(LCDM)
#test_get_subhalo_part_idx(LCDM)
