import sys,os 
import numpy as np
import h5py
from collections import Counter
import matplotlib.pyplot as plt

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
    v1kpc = dataset.get_subhalos(attr)
    vmax = dataset.get_subhalos('Vmax')
    print(np.median(v1kpc))
    print(np.median(vmax))
    print(np.min(v1kpc),np.max(v1kpc))
    print(np.min(vmax),np.max(vmax))

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

def test_get_subhalos_IDs_type(snapshot):

    # Output should be an ndarray of lists of np.uint64:
    for pt in [[0],[1],[0,1],[4],[5],[0,1,4,5],[]]:
        print("part_type = {}".format(pt))
        IDs = snapshot.get_subhalos_IDs(part_type=pt)
        print(type(IDs))
        print(IDs.shape)
        print(type(IDs[0]))
        for el in (IDs[0] + IDs[140] + IDs[-23]):
            if type(el) != np.uint64:
                print('problem')
        print('')

def test_get_subhalos_IDs_ptSelection(snapshot):

    gas = snapshot.get_subhalos_IDs(part_type=[0])
    dm = snapshot.get_subhalos_IDs(part_type=[1])
    combined = snapshot.get_subhalos_IDs(part_type=[0,1])

    print('len(gas[0])=',len(gas[0]))
    print('len(dm[0])=',len(dm[0]))
    print('len(combined[0])=',len(combined[0]))
    intersection = list(set(dm[0]) & set(combined[0]))
    print('len(intersection)=',len(intersection))

    n = 20
    sel = np.random.choice(np.arange(dm.size),n)
    for i in sel:
        intersection = list(set(dm[i]) & set(combined[i]))
        if (len(intersection) != len(dm[i])):
            print('problem: len(dm[{}]) = {}, len(intersection) = {}'\
                    .format(i,len(dm[i]),len(intersection)))

def test_get_subhalos_IDs_pt(snapshot):
    mask = (snapshot.get_subhalos('GroupNumber') == 1)
    pt = [0]
    SGNs = snapshot.get_subhalos('SubGroupNumber')[mask]
    IDs = snapshot.get_subhalos_IDs(part_type=pt)[mask]
    IDs_from_sd = snapshot.get_all_bound_IDs()
    IDs_from_pd = snapshot.get_particles("ParticleIDs",part_type=pt)
    IDs_from_pd_all = snapshot.get_particles("ParticleIDs")
    n = 20
    sel = np.random.choice(np.arange(IDs.size),n)
    sel = np.where(SGNs == 0)[0]
    for i in sel:
        intersection = np.intersect1d(\
                np.array(IDs[i]),IDs_from_pd,assume_unique=True)
        if (len(intersection) != len(IDs[i])):
            print('problem: len(IDs[{}]) = {}, len(from_pd) = {},'\
                    .format(i,len(IDs[i]),len(IDs_from_pd)),\
                    'len(intersection) = {}'.format(len(intersection)))
            for p in [e for e in range(6) if e not in pt]:
                compare = snapshot.get_particles("ParticleIDs",\
                        part_type=[p])
                intersection = np.intersect1d(\
                        np.array(IDs[i]),compare,assume_unique=True)
                print('intersection with pt={}: {},'\
                            .format(p,len(intersection)))

#            inters,j,idxs = np.intersect1d(np.array(IDs[i]),\
#                    IDs_from_pd_all,assume_unique=True,\
#                    return_indices=True)
#            inters2,j,idxs2 = np.intersect1d(np.array(IDs[i]),\
#                    IDs_from_sd, assume_unique=True,\
#                    return_indices=True)
#
#            idxs2 = np.sort(idxs2)
#            print(idxs2[0],idxs2[-1],len(idxs2))
#            for i in range(len(idxs2)-1):
#                if idxs2[i+1]-idxs2[i] != 1:
#                    print(idxs2[i],idxs2[i+1])


#            mass_form_pd = snapshot.get_particle_masses(part_type=pt)
#            mass_form_pd_all = snapshot.get_particle_masses()
#            print(np.min(mass_form_pd),np.max(mass_form_pd))
#            print(len(inters))
#            print(np.min(mass_form_pd_all[idxs]),\
#                    np.max(mass_form_pd_all[idxs]))
#            cnts = Counter(mass_form_pd_all[idxs])
#            for item in cnts.items():
#                print(item)
#
#            # most frequent is almost without exception the pt=1 mass:
#            most_freq = cnts.most_common(1)[0][0]
#            odd_ones = np.argwhere([m != most_freq for m in \
#                    mass_form_pd_all[idxs]])
#            print('idxs of odd ones:', odd_ones)
##            print("min and max idx of odds:",\
##                    np.min(odd_ones),np.max(odd_ones))
#            print('-----------------\n')

def test_IDs_dist(snapshot):
    GNs = snapshot.get_subhalos('GroupNumber')
    mask = (GNs == 1)
    SGNs = snapshot.get_subhalos('SubGroupNumber')[mask]
    pt = [0]

    IDs = snapshot.get_subhalos_IDs(part_type=pt)[mask]
    COPs = snapshot.get_subhalos("CentreOfPotential")[mask]

    # Get box size:
    with h5py.File(snapshot.part_file,'r') as partf:
        h = partf['link1/Header'].attrs.get('HubbleParam')
        boxs = partf['link1/Header'].attrs.get('BoxSize') * 1000/h 
                                                            # Mpc/h -> kpc

    IDs_from_pd_all = snapshot.get_particles("ParticleIDs")
    coords_all = snapshot.get_particles("Coordinates")
    n = 10
    #sel = np.random.choice(np.arange(IDs.size),n)
    sel = np.where(SGNs == 0)[0]
    print(sel)
    for i in sel:
        cop = COPs[i]
        inters,j,idxs = np.intersect1d(np.array(IDs[i]),\
                IDs_from_pd_all,assume_unique=True,\
                return_indices=True)
        coords_sel = coords_all[idxs]

        # Calculate distances to cop:
        d = np.mod(coords_sel-cop+0.5*boxs, boxs) - 0.5*boxs
        r = np.linalg.norm(d, axis=1)
        fig,ax = plt.subplots()
        ax.hist(np.log(r))
        plt.show()


def test_get_all_bound_IDs(snapshot):
    IDs = snapshot.get_particles('ParticleIDs')
    GNs = snapshot.get_particles('GroupNumber')
    IDs_bound = IDs[GNs < 10**10]
    test_bound = snapshot.get_all_bound_IDs()
    print(test_bound.size)
    print(test_bound[0])
    intersection = np.intersect1d(test_bound,IDs_bound)
    if intersection.size == test_bound.size:
        print('success')
    else:
        print('failure')

def test_get_subhalos_IDs(snapshot):
    IDs = snapshot.get_subhalos_IDs(part_type=[0,4])
#    print('s',IDs.shape)
#    print(IDs[0].shape)
#    print(IDs[-1].shape)
#    print(IDs[1].shape)
#    IDs = snapshot.get_subhalos_IDs(list(range(65,67)))
#    print(len(IDs))
#    print(IDs[0].shape)
#    print(IDs[-1].shape)
#    print(IDs[1].shape)

#    print('----------------')
#    IDs = snapshot.get_subhalos_IDs(part_type=[0,1])
#    print('----------------')
#    offs = snapshot.get_subhalos('SubOffset')
#    subLT = snapshot.get_subhalos('SubLengthType')
#    print(IDs.shape)
#    for i,x in enumerate(zip(IDs[:50],subLT[:50])):
#        ids = x[0]; slt = x[1]
#        print(i,len(ids))
#        print(slt)

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
        keys,sorting = snap.link_select('group',fnums)
        print(keys)
        print(sorting)

    selections = [[],list(range(10))]
    for fnums in selections:
        keys,sorting = snap.link_select('particle',fnums)
        print(keys)
        print(sorting)

def test_get_subhalos_order(snap):
    gns = snap.get_subhalos('GroupNumber')
    sgns = snap.get_subhalos('SubGroupNumber')
    for gn,sgn in zip(gns,sgns):
        print(gn,sgn)

def test_peculiar_files():
    snap = Snapshot("CDM_V1_LR",115)
    snap.get_particles('Coordinates', part_type=[5])

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
#test_get_particle_masses(LCDM)
#test_convert_to_cgs_part(LCDM)
#test_file_of_halo(LCDM)
#test_order_of_links(LCDM)
#test_get_subhalos_with_fnums(LCDM)
#test_get_subhalos_IDs_single(LCDM)
#test_get_subhalos_IDs(LCDM)
#test_get_subhalos_IDs_ptSelection(LCDM)
#test_get_subhalos_IDs_pt(LCDM)
test_get_all_bound_IDs(LCDM)
#test_IDs_dist(LCDM)
#test_link_select(LCDM)
#test_get_subhalos_order(LCDM)
#test_peculiar_files()

#sgns = LCDM.get_subhalos('SubGroupNumber',False)[0]
#print(sgns.size)
#print(sgns[63:68])
