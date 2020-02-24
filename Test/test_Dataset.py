import sys,os 
import h5py

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from dataset import Dataset

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

def test_get_subhalos(dataset):
    attr = "Vmax"
    print(len(dataset.get_subhalos(attr, False)[0]))
    sat, isol = dataset.get_subhalos(attr, True)
    print(len(sat))
    print(len(isol))


LCDM = Dataset("V1_LR_fix_127_z000p000","LCDM")
#test_get_data_path(LCDM)
#test_count_files(LCDM)
#test_make_group_file(LCDM)
test_read_subhalo_attr(LCDM)
test_get_subhalos(LCDM)
