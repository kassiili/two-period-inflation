import numpy as np

import snapshot_obj


def inspect_groupnumbers(snap):
    gns = snap.get_particles('GroupNumber')
    sgns = snap.get_particles('SubGroupNumber')
    inf = np.max(gns)

    print(np.unique(sgns))
    # Possible values of sgn:
    #       1) 0 < gn < inf: particles bound to a subhalo
    #       2) gn == inf: particles notbound to a subhalo

    print(np.unique(gns))
    print(np.sum(gns == 0))
    # Possible values of gn:
    #       1) 0 < gn < inf: particles bound to FOF group
    #               Possible values of sgn:
    #                   1.1) sgn < inf: particles bound to a subhalo (or
    #                   the central halo)
    #                   1.2) sgn == inf, the "fuzz": particles
    #                   identified with the FOF halo but not bound to it
    #                   (thus not having sgn == 0)
    #               - NOTE: no particles with gn == 0 !!!
    #       2) gn == inf: free particles not identified with any FOF
    #       group and thus not with any subhalo
    #       3) gn < 0: ?????

    # Case 1)
    mask_regular_gn = np.logical_and(gns > 0, gns < inf)
    mask_bound = np.logical_and.reduce((mask_regular_gn, sgns < inf))
    print("case 1) #", np.sum(mask_regular_gn))
    print("case 1.1) #", np.sum(mask_bound))
    print("case 1) possible gn values ", np.unique(gns[mask_regular_gn]))
    print("case 1) possible sgn values ", np.unique(sgns[
                                                         mask_regular_gn]))

    # Case 2)
    mask_unidentified = (gns == inf)
    print(np.sum(mask_unidentified))
    print(np.unique(gns[mask_unidentified]))
    print(np.unique(sgns[mask_unidentified]))

    # Case 3)
    mask_wtf = (gns < 0)
    print(np.sum(gns < 0))


sim_id = "V1_LR_fix"
inspect_groupnumbers(snapshot_obj.Snapshot(sim_id, 127))
