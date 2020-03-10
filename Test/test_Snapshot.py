import sys,os 
import numpy as np
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snapshot_obj import Snapshot
from curve_fit import poly_fit

def test_get_data_path(dataset):
    print(dataset.get_data_path("part"))
    print(dataset.get_data_path("group"))
    print(os.listdir(os.path.join("..","PlotSubhaloData",
        dataset.get_data_path("part"))))

def test_count_files(dataset):
    print(dataset.count_files())

def test_make_group_file(dataset):
    filename = dataset.grp_file
    print(filename)
    with h5py.File(filename,'r') as grpf:
        print("# of links:",len(grpf.keys()))
        print("link1 contents:",grpf['link1'].keys())
        print("link1/Subhalo contents:",grpf['link1/Subhalo'].keys())
        print("link78/Subhalo/Vmax contents:",grpf['link78/Subhalo/Vmax'])

def test_read_subhalo_attr(dataset):
    attr = 'Vmax'
    print(len(dataset.read_subhalo_attr(attr)))
    attr = 'Stars/Mass'
    print(len(dataset.read_subhalo_attr(attr)))

def test_get_subhalos(dataset):
    attr = "Vmax"
    print(type(dataset.get_subhalos('GroupNumber',False)[0]))
    print(type(dataset.get_subhalos(attr)[0]))
    print(len(dataset.get_subhalos(attr, False)[0]))
    sat, isol = dataset.get_subhalos(attr, True)
    print(len(sat))
    print(len(isol))

def test_make_part_file(dataset):
    filename = dataset.part_file
    print(filename)
    with h5py.File(filename,'r') as partf:
        print("# of links:",len(partf.keys()))
        print("link0 contents:",partf['link0'].keys())
#        print("link1 contents:",partf['link1'].keys())
#        print("link6 contents:",partf['link6'].keys())
#        for i in [0,1,2,3,4,5]:
#            print("link1/PartType{} contents:".format(i),\
#                    partf['link1/PartType{}'.format(i)].keys())
#        print("link12/PartType1/SubGroupNumber contents:",\
#                partf['link12/PartType1/SubGroupNumber'])

def test_get_particles(dataset):
    attr = "Coordinates"
    data = dataset.get_particles(attr)
    print(len(data))
    print(data[:10])

def test_calcVelocitiesAt1kpc(dataset):
    v1kpc = dataset.calcVelocitiesAt1kpc()
    sgns = dataset.get_subhalos('SubGroupNumber', divided=False)[0]
    print(v1kpc.size)
    print(sgns.size)

def test_get_subhalos_V1kpc(dataset):
    attr = "V1kpc"
    v1kpc = dataset.get_subhalos(attr,divided=False)
    vmax = dataset.get_subhalos('Vmax',divided=False)
    print(v1kpc)
    print(vmax)
    print(np.min(v1kpc),np.max(v1kpc))

def test_get_subhalo_part_idx(dataset):
    idx = dataset.get_subhalo_part_idx()
    i=0
    for l in idx:
        if sum(l) != 0:
            print(sum(l))
            i+=1
    print(i)
    
def test_group_name(dataset):
    filename = dataset.grp_file
    with h5py.File(filename,'r') as grpf:
        for (name,f) in grpf.items():
            if ('link' in name):
                print(name)
    
def test_gn_counts(dataset):
#    sat, isol = dataset.get_subhalos('GroupNumber')
#    print(np.bincount(sat.astype(int)))
#    print(sum(sat))
#    print(np.bincount(isol.astype(int)))

    a = dataset.get_subhalos('GroupNumber',divided=False)[0].astype(int)
    cnt = np.bincount(a)
    print(cnt)
    print(sum(cnt==0))

def test_get_particle_masses(dataset):
    masses = dataset.get_particle_masses()
    print(masses[:100])
    print(min(masses),max(masses)) 
    print(sum(masses==0)) # compare to type 2,3 part numbers

def test_convert_to_cgs_part(dataset):
    gns = dataset.get_particles('GroupNumber')
    print(len(gns))
        
def test_file_of_halo(dataset):
    for (gn,sgn) in [(1,0),(1,45),(2,0),(12,0),(123,0),(3410,0)]:
        print((gn,sgn),dataset.file_of_halo(gn,sgn))

def test_order_of_links(dataset):
    return None

LCDM = Snapshot("CDM_V1_LR",127,"LCDM")
#LCDM_x = Snapshot("CDM_V1_LR", 101, "LCDM")
#test_get_data_path(LCDM_x)
#test_count_files(LCDM)
#test_make_group_file(LCDM)
#test_read_subhalo_attr(LCDM)
#test_get_subhalos(LCDM)
#test_make_part_file(LCDM)
#test_get_particles(LCDM)
#test_calcVelocitiesAt1kpc(LCDM)
#test_get_subhalos_V1kpc(LCDM)
#test_get_subhalo_part_idx(LCDM)
#test_group_name(LCDM)
#test_gn_counts(LCDM)
#test_get_particle_masses(LCDM)
#test_convert_to_cgs_part(LCDM)
#test_file_of_halo(LCDM)
#test_order_of_links(LCDM)
