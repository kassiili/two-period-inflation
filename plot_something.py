from PlotSubhaloData import plot_SM_vs_Vmax, dataset #, calc_median.calc_median_trend

plot = plot_SM_vs_Vmax.plot_SM_vs_Vmax(satellites=1)
LCDM_dataset = dataset.dataset(dir='V1_MR_fix_082_z001p941', name='LCDM', nfiles_part=16, nfiles_group=192)
LCDM = plot_SM_vs_Vmax.SM_vs_Vmax_data(LCDM_dataset)
curvaton_dataset = dataset.dataset(dir='V1_MR_mock_1_fix_082_z001p941', name='curvaton', nfiles_part=1, nfiles_group=64)
curvaton = plot_SM_vs_Vmax.SM_vs_Vmax_data(curvaton_dataset)

plot.add_data(LCDM, ['o','pink','red'])
plot.add_data(curvaton, ['o','lightblue','blue'])
plot.save_figure('') 
