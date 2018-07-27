from dataset import dataset
from plot_SM_vs_Vmax import plot_SM_vs_Vmax, SM_vs_Vmax_data
from plot_cumulDistByVmax import plot_subhalo_dist_vs_vmax, subhalo_dist_vs_vmax_data
from plot_Rmax_vs_Vmax import plot_rmax_vs_vmax, rmax_vs_vmax_data
from plot_rotation_curve import plot_rotation_curve, rotation_curve

LCDM_dataset = dataset('V1_MR_fix_082_z001p941', 'LCDM', 16, 192)
mock_dataset = dataset('V1_MR_mock_1_fix_082_z001p941', 'mock', 1, 64)

#plot = plot_SM_vs_Vmax()
#plot.add_data(SM_vs_Vmax_data(LCDM_dataset), 1, 'red')
#plot.add_data(SM_vs_Vmax_data(mock_dataset), 1, 'blue')

#plot = plot_subhalo_dist_vs_vmax()
#plot.add_data(subhalo_dist_vs_vmax_data(LCDM_dataset), 1, ['lightblue', 'blue'])
#plot.add_data(subhalo_dist_vs_vmax_data(mock_dataset), 1, ['pink', 'red'])

#plot = plot_rmax_vs_vmax()
#plot.add_data(rmax_vs_vmax_data(LCDM_dataset), 1, 'red')
#plot.add_data(rmax_vs_vmax_data(mock_dataset), 1, 'blue')

gn = 1; sgn = 0
plot = plot_rotation_curve(gn, sgn)
plot.add_data(rotation_curve(gn, sgn, LCDM_dataset))
plot.add_data(rotation_curve(gn, sgn, mock_dataset))

plot.save_figure() 
    
