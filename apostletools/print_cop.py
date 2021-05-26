import numpy as np
import sys, getopt

import snapshot_obj

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hg:s:",\
                ["ge=","sgn="])
    except getopt.GetoptError:
        print('test.py -g <group number> -s <subgroup number>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -g <group number> -s <subgroup number>')
            sys.exit()
        elif opt in ("-g", "--gn"):
            gn = int(arg)
        elif opt in ("-s", "--sgn"):
            sgn = int(arg)
    
    snap = snapshot_obj.Snapshot("CDM_V1_LR",127)
    
    gns = snap.get_subhalos("GroupNumber")
    sgns = snap.get_subhalos("SubGroupNumber")
    COP = snap.get_subhalos("CentreOfPotential",units='')\
            [np.logical_and(gns==gn,sgns==sgn)]
    

    print("Centre of potential in units Mpc/h:\n    {}".format(\
            COP))

if __name__ == "__main__":
   main(sys.argv[1:])

