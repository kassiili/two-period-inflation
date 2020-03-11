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
        idx, data = trace_halo.get_subhalo(snap,attr,gn,sgn)
        print(idx,data)
        print("Correct:",gn,sgn)
        g = snap.get_subhalos('GroupNumber',False)[0][idx]
        s = snap.get_subhalos('SubGroupNumber',False)[0][idx]
        print("Returned:",g,s)

def test_get_subhalo_IDs(snap,gn,sgn):
    idx, data = trace_halo.get_subhalo_IDs(snap,gn,sgn)
    print(idx,data)
    print("Correct:",gn,sgn)
    g = snap.get_subhalos('GroupNumber',False)[0][idx]
    s = snap.get_subhalos('SubGroupNumber',False)[0][idx]
    print("Returned:",g,s)

LCDM = Snapshot("CDM_V1_LR",127,"LCDM")
#test_trace_halo(LCDM)
#test_get_subhalo(LCDM,2,8)
test_get_subhalo_IDs(LCDM,2,8)
