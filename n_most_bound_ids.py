import numpy as np
import sys, getopt

import snapshot_obj

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hg:s:p:n:",\
                ["ge=","sgn=","pt=","mostbound="])
    except getopt.GetoptError:
        print('test.py -g <group number> -s <subgroup number> -p ' + \
                '<particle type> -n <# of most bound>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -g <group number> -s <subgroup number> -p ' + \
                '<particle type> -n <# of most bound>')
            sys.exit()
        elif opt in ("-g", "--gn"):
            gn = int(arg)
        elif opt in ("-s", "--sgn"):
            sgn = int(arg)
        elif opt in ("-p", "--pt"):
            pt = int(arg)
        elif opt in ("-n", "--mostbound"):
            n = int(arg)
    
    print(gn,sgn,pt,n)
    snap = snapshot_obj.Snapshot("CDM_V1_LR",127)
    
    gns = snap.get_subhalos("GroupNumber")
    sgns = snap.get_subhalos("SubGroupNumber")
    IDs = snap.get_subhalos_IDs(part_type=pt)
    
    # Select halo:
    IDs = IDs[np.logical_and((gn==gns),(sgn==sgns))][0]
    
    # Get n most bound:
    mostBound = IDs[:n]
    
    np.array(mostBound).tofile("ids_for_gadgetviewer.dat",sep="\n")

if __name__ == "__main__":
   main(sys.argv[1:])

