import sys,os 
import numpy as np
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snapshot_obj import Snapshot
from curve_fit import poly_fit
import dataset_compute

def test_split_satellites(snap):
    attr = "Vmax"
    sat_GN,isol_GN = dataset_compute.split_satellites(snap,'GroupNumber')
    print(type(sat_GN))
    print(len(sat_GN),len(isol_GN))
    sat, isol = dataset_compute.split_satellites(snap,\
            attr,fnums=list(range(4)))
    print(len(sat))
    print(isol[:25])

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

def test_trace_halo(dataset):
    print(dataset_compute.trace_halo(dataset,2,8))

LCDM = Snapshot("CDM_V1_LR",127,"LCDM")
test_split_satellites(LCDM)
#test_calculate_V1kpc(LCDM)
#test_get_subhalo_part_idx(LCDM)
#test_trace_halo(LCDM)
