import sys,os 

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from PlotSubhaloData.dataset import Dataset

def test_get_data_path(dataset):
    print(dataset.get_data_path("part"))
    print(dataset.get_data_path("group"))
    print(os.listdir(os.path.join("..","PlotSubhaloData",
        dataset.get_data_path("part"))))

def test_count_files(dataset):
    print(dataset.count_files())

LCDM = Dataset("V1_LR_fix_127_z000p000","LCDM")
test_get_data_path(LCDM)
test_count_files(LCDM)
