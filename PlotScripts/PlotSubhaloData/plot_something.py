from dataset import dataset
from plot_SM_vs_Vmax import plot_SM_vs_Vmax, SM_vs_Vmax_data
from plot_cumulDistByVmax import plot_subhalo_dist_vs_vmax, subhalo_dist_vs_vmax_data
from plot_Rmax_vs_Vmax import plot_rmax_vs_vmax, rmax_vs_vmax_data
from plot_rotation_curve import plot_rotation_curve, rotation_curve
from plot_Vmax_vs_V1kpc import plot_Vmax_vs_V1kpc, Vmax_vs_V1kpc_data

LCDM_dataset = dataset('V1_MR_fix_082_z001p941', 'LCDM', 16, 192)
mock_dataset = dataset('V1_MR_mock_1_fix_082_z001p941', 'mock', 1, 64)
LR_CDM_dataset = dataset('V1_LR_fix_127_z000p000', 'LR', 16, 96)
LCDM_z0_dataset = dataset('V1_MR_fix_127_z000p000', 'z=0', 16, 96)
LCDM_z2_dataset = dataset('V1_MR_fix_082_z001p941', 'z~2', 16, 192)

#plot = plot_SM_vs_Vmax(1)
#plot.add_data(SM_vs_Vmax_data(LCDM_z0_dataset), 'red')
#plot.add_data(SM_vs_Vmax_data(LCDM_z2_dataset), 'blue')

plot = plot_subhalo_dist_vs_vmax(1)
plot.add_data(subhalo_dist_vs_vmax_data(LCDM_z0_dataset), ['lightblue', 'blue'])
plot.add_data(subhalo_dist_vs_vmax_data(LCDM_z2_dataset), ['pink', 'red'])

#plot = plot_rmax_vs_vmax(1)
#plot.add_data(rmax_vs_vmax_data(LCDM_z0_dataset), 'red')
#plot.add_data(rmax_vs_vmax_data(LCDM_z2_dataset), 'blue')

#gn = 1; sgn = 0
#plot = plot_rotation_curve(gn, sgn)
#plot.add_data(rotation_curve(gn, sgn, LCDM_dataset))
#plot.add_data(rotation_curve(gn, sgn, mock_dataset))

#plot = plot_Vmax_vs_V1kpc(1)
#plot.add_data(Vmax_vs_V1kpc_data(LR_CDM_dataset), 'red')
#plot.add_data(Vmax_vs_V1kpc_data(LCDM_dataset), 'red')
#plot.add_data(Vmax_vs_V1kpc_data(mock_dataset), 'blue')

#plot.save_figure('Comparisons_082_z001p941') 
plot.save_figure('Comparisons_CDM_z0_vs_z2')
