import numpy as np
import os, sys, getopt

sys.path.append('..')
import snapshot_obj

def main(argv):
    """ Retrieve the n most bound particles 
        
        ...of given type (-p)
        ...of a given halo (-g -s)
        ...in a given snapshot (or time) (-t)
        ...of a given simulation (or model) (-m).
    """
    fname = os.path.basename(__file__)
    try:
        opts, args = getopt.getopt(argv,"hg:s:p:n:t:m:",\
                ["ge=","sgn=","pt=","mostbound=","snapshot=","simulation="])
    except getopt.GetoptError:
        print('python {} -g <group number> \n'.format(fname) + \
                '   -s <subgroup number> \n' + \
                '   -p <particle type> \n' + \
                '   -n <# of most bound> \n' + \
                '   -t <snapshot id> \n' + \
                '   -m <simulation id>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python {} -g <group number> \n'.format(fname) + \
                    '   -s <subgroup number> \n' + \
                    '   -p <particle type> \n' + \
                    '   -n <# of most bound> \n' + \
                    '   -t <snapshot id> \n' + \
                    '   -m <simulation id>')
            sys.exit()
        elif opt in ("-g", "--gn"):
            gn = int(arg)
        elif opt in ("-s", "--sgn"):
            sgn = int(arg)
        elif opt in ("-p", "--pt"):
            pt = int(arg)
        elif opt in ("-n", "--mostbound"):
            n = int(arg)
        elif opt in ("-t", "--snapshot"):
            snap = int(arg)
        elif opt in ("-m", "--simulation"):
            sim = arg
    
    print(gn,sgn,pt,n,snap,sim)
    snap = snapshot_obj.Snapshot(sim,snap)
    
    gns = snap.get_subhalos("GroupNumber")
    sgns = snap.get_subhalos("SubGroupNumber")
    IDs = snap.get_subhalos_IDs(part_type=pt)
    
    # Select halo:
    IDs = IDs[np.logical_and((gn==gns),(sgn==sgns))][0].astype(int)
    
    # Get n most bound:
    mostBound = IDs[:n]
    
    np.array(mostBound).tofile("ids_for_gadgetviewer.dat",sep="\n")

if __name__ == "__main__":
   main(sys.argv[1:])

