from PlotSubhaloData import plot_SM_vs_Vmax #, calc_median.calc_median_trend

plot = plot_SM_vs_Vmax()
LCDM = SM_vs_Vmax_data(dataset='V1_MR_fix_082_z001p941', nfiles_part=16, nfiles_group=192)
curvaton = SM_vs_Vmax_data(dataset='V1_MR_mock_1_fix_082_z001p941', nfiles_part=1, nfiles_group=64)

plot.add_data(LCDM, 1, 'red')
plot.add_data(curvaton, 1, 'blue')
plot.save_figure() 
