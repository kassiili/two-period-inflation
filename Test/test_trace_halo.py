import sys,os 
import numpy as np
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapshot_obj import Snapshot
from curve_fit import poly_fit
import trace_halo

def test_trace_halo(snap):
    print(trace_halo.trace_halo(snap,2,8))

def test_get_subhalo(snap,gn,sgn):
    attrs = ['GroupNumber','Vmax']
    for attr in attrs:
        print(trace_halo.get_subhalo(snap,attr,gn,sgn))

LCDM = Snapshot("CDM_V1_LR",127,"LCDM")
#test_trace_halo(LCDM)
test_get_subhalo(LCDM,2,8)
