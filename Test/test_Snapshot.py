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
    attr = "SubLengthType"
    print(type(dataset.get_subhalos('GroupNumber')))
    data = dataset.get_subhalos(attr)
    print(data.shape)
    print(len(data))
    print(data[:20,1])

def test_get_subhalos_SubLengthType(snap):
    attr = "SubLengthType"
    data = snap.get_subhalos(attr)
    print(data.shape)
    print(data[:,0].shape)

def test_get_subhalos_with_fnums(dataset):
    attr = "Vmax"
    print(len(dataset.get_subhalos(attr)))
    print(len(dataset.get_subhalos(attr,range(6))))

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
    for pt in [0,1,4,5]:
        masses = dataset.get_particle_masses(part_type=[pt])
        print(masses[:30])
#    masses = dataset.get_particle_masses()
#    IDs = dataset.get_particles('ParticleIDs')
#    print(masses.shape)
#    print(IDs.size)
#    print(masses[:20])
#    print(min(masses),max(masses)) 
#    print(sum(masses==0)) # compare to type 2,3 part numbers

def test_convert_to_cgs_part(dataset):
    gns = dataset.get_particles('GroupNumber')
    print(len(gns))
        
def test_file_of_halo(dataset):
    halos =\
    [(1,0),(1,45),(2,0),(2,9),(2,55),(3,10),(12,0),(123,0),(862,0)]
    #halos = [(123,0)]
    for (gn,sgn) in halos:
        fnum = dataset.file_of_halo(gn,sgn)
        print((gn,sgn),fnum)
        gns = dataset.get_subhalos('GroupNumber',False,fnums=[fnum])[0]
        sgns = dataset.get_subhalos('SubGroupNumber',False,fnums=[fnum])[0]
        #print(dataset.file_of_halo(gns[1],sgns[1]))
        print(np.argwhere(np.logical_and((gns==gn),(sgns==sgn))))

def test_order_of_links(dataset):
    return None

def test_get_subhalos_IDs_single(dataset):
    IDs = dataset.get_subhalos_IDs(list(range(5)))
    print(IDs)
    print(type(IDs[0]))
    for el in IDs:
        print(el.shape)

def test_get_subhalos_IDs(dataset):
    IDs = dataset.get_subhalos_IDs()
    print('s',IDs.shape)
    print(IDs[0].shape)
    print(IDs[-1].shape)
    print(IDs[1].shape)
    IDs = dataset.get_subhalos_IDs(list(range(65,67)))
    print(len(IDs))
    print(IDs[0].shape)
    print(IDs[-1].shape)
    print(IDs[1].shape)
    #print(IDs)

def test_get_subhalos_IDs_DMO(snap):
    fnum=0
    IDs = snap.get_subhalos_IDs_DMO([fnum])
    subLengthType = snap.get_subhalos('SubLengthType',fnums=[fnum])
    print(subLengthType[:20].astype(int))

    matching_lengths = [len(ids) == typelen[1] for (ids,typelen) in\
        zip(IDs,subLengthType)]

    if sum(matching_lengths) == len(IDs):
        print("lengths match")
    else:
        print("lengths no match")

    from_part_file =\
            snap.get_particles('ParticleIDs',part_type=[1,2,4,5]).astype(int)
    print(IDs[0])
    print(from_part_file.size)
    print(IDs.shape)
    for ids in IDs:
        print(len(ids))
        shared = np.intersect1d(ids,from_part_file)
        print(len(shared))
        print(len(ids)==len(shared))

def contained(arr1,arr2):
    arr2 = set(arr2)
    contained = [elem in arr2 for elem in arr1]
    print(sum(contained))
    if sum(contained) == len(arr1):
        return True
    return False

def test_link_select(snap):
    selections = [[],list(range(45))]
    for fnums in selections:
        keys,sorting = snap.link_select(fnums)
        print(keys)
        print(sorting)

def test_get_subhalos_order(snap):
    gns = snap.get_subhalos('GroupNumber')
    sgns = snap.get_subhalos('SubGroupNumber')
    for gn,sgn in zip(gns,sgns):
        print(gn,sgn)

LCDM = Snapshot("CDM_V1_LR",127,"LCDM")
#LCDM_x = Snapshot("CDM_V1_LR", 101, "LCDM")
#test_get_data_path(LCDM_x)
#test_count_files(LCDM)
#test_make_group_file(LCDM)
#test_read_subhalo_attr(LCDM)
#test_get_subhalos(LCDM)
#test_get_subhalos_SubLengthType(LCDM)
#test_make_part_file(LCDM)
#test_get_particles(LCDM)
#test_calcVelocitiesAt1kpc(LCDM)
#test_get_subhalos_V1kpc(LCDM)
#test_get_subhalo_part_idx(LCDM)
#test_group_name(LCDM)
#test_gn_counts(LCDM)
test_get_particle_masses(LCDM)
#test_convert_to_cgs_part(LCDM)
#test_file_of_halo(LCDM)
#test_order_of_links(LCDM)
#test_get_subhalos_with_fnums(LCDM)
#test_get_subhalos_IDs_single(LCDM)
#test_get_subhalos_IDs(LCDM)
#test_get_subhalos_IDs_DMO(LCDM)
#test_link_select(LCDM)
#test_get_subhalos_order(LCDM)

#sgns = LCDM.get_subhalos('SubGroupNumber',False)[0]
#print(sgns.size)
#print(sgns[63:68])
