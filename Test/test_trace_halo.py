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
        data,idx = trace_halo.get_subhalo(snap,attr,gn,sgn)
        print(idx,data)
        print("Correct:",gn,sgn)
        g = snap.get_subhalos('GroupNumber',False)[0][idx]
        s = snap.get_subhalos('SubGroupNumber',False)[0][idx]
        print("Returned:",g,s)

def test_get_subhalo_IDs(snap,gn,sgn):
    data, idx = trace_halo.get_subhalo_IDs(snap,gn,sgn)
    print(data.shape)
    print(type(data))
#    print(idx,data)
    print("Correct:",gn,sgn)
    g = snap.get_subhalos('GroupNumber',False)[0][idx]
    s = snap.get_subhalos('SubGroupNumber',False)[0][idx]
    print("Returned:",g,s)

def test_match_halo():

    snap1 = Snapshot("CDM_V1_LR",127,"LCDM")    # Reference
    snap2 = Snapshot("CDM_V1_LR",127,"LCDM")    # Same
    snap3 = Snapshot("CDM_V1_LR",127,"LCDM")    # Check that only one
                                                # match
    snap4 = Snapshot("CDM_V1_LR",126,"LCDM")    # Most probable match
    snap5 = Snapshot("CDM_V1_LR",125,"LCDM")    # Certainly not a match
    snap6 = Snapshot("CDM_V1_LR",120,"LCDM")    # Trying our luck

    gn1 = 2; sgn1 = 8
    gn2 = 2; sgn2 = 8
    gn3 = 2; sgn3 = 7
    gn4 = 2; sgn4 = 8
    gn5 = 20; sgn5 = 0
    gn6 = 2; sgn6 = 8
    
    # Get particle IDs and halo mass:
    IDs1,_ = trace_halo.get_subhalo_IDs(snap1,gn1,sgn1)
    mass1 = trace_halo.get_subhalo(snap1,'Mass',gn1,sgn1)[0]
    IDs2,_ = trace_halo.get_subhalo_IDs(snap2,gn2,sgn2)
    mass2 = trace_halo.get_subhalo(snap2,'Mass',gn2,sgn2)[0]
    IDs3,_ = trace_halo.get_subhalo_IDs(snap3,gn3,sgn3)
    mass3 = trace_halo.get_subhalo(snap3,'Mass',gn3,sgn3)[0]
    IDs4,_ = trace_halo.get_subhalo_IDs(snap4,gn4,sgn4)
    mass4 = trace_halo.get_subhalo(snap4,'Mass',gn4,sgn4)[0]
    IDs5,_ = trace_halo.get_subhalo_IDs(snap5,gn5,sgn5)
    mass5 = trace_halo.get_subhalo(snap5,'Mass',gn5,sgn5)[0]
    IDs6,_ = trace_halo.get_subhalo_IDs(snap6,gn6,sgn6)
    mass6 = trace_halo.get_subhalo(snap6,'Mass',gn6,sgn6)[0]

    print(trace_halo.match_subhalos(IDs1,mass1,IDs2,mass2))
    print(trace_halo.match_subhalos(IDs1,mass1,IDs3,mass3))
    print(trace_halo.match_subhalos(IDs1,mass1,IDs4,mass4))
    print(trace_halo.match_subhalos(IDs1,mass1,IDs5,mass5))
    print(trace_halo.match_subhalos(IDs1,mass1,IDs6,mass6))

def test_match_snapshots():

    snap_ref = Snapshot("CDM_V1_LR",127,"LCDM")    # Reference
    snap = Snapshot("CDM_V1_LR",126,"LCDM")

    gns = snap_ref.get_subhalos('GroupNumber')
    sgns = snap_ref.get_subhalos('SubGroupNumber')
    print(np.bincount(gns.astype(int))[:15])
    print(np.bincount(sgns.astype(int))[:15])

    gns = snap.get_subhalos('GroupNumber')
    sgns = snap.get_subhalos('SubGroupNumber')
    print(np.bincount(gns.astype(int))[:15])
    print(np.bincount(sgns.astype(int))[:15])

    gn = 2; sgn = 4
    gn_m, sgn_m = trace_halo.match_snapshots(snap, snap_ref, gn, sgn)
    ids,_ = trace_halo.get_subhalo_IDs(snap_ref, gn, sgn)
    ids_m,_ = trace_halo.get_subhalo_IDs(snap, gn_m, sgn_m)
    m,_ = trace_halo.get_subhalo(snap_ref, 'Mass', gn, sgn)
    m_m,_ = trace_halo.get_subhalo(snap, 'Mass', gn_m, sgn_m)

    print(trace_halo.match_subhalos(ids,m,ids_m,m_m))
    print(trace_halo.get_subhalo(snap_ref,'Stars/Mass',gn,sgn))
    print(trace_halo.get_subhalo(snap,'Stars/Mass',gn_m,sgn_m))

LCDM = Snapshot("CDM_V1_LR",127,"LCDM")
#test_trace_halo(LCDM)
#test_get_subhalo(LCDM,2,8)
#test_get_subhalo_IDs(LCDM,2,8)
#test_match_halo()
test_match_snapshots()
