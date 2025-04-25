from matplotlib.colors import LinearSegmentedColormap

colors = ['white', '#ff6347']
n_bins = 256
orange_cmap1 = LinearSegmentedColormap.from_list('custom_orange', colors, N=n_bins)
orange_cmap2 = LinearSegmentedColormap.from_list('custom_orange', [(1, 1, 1), (1, 0.388, 0.278)], N=n_bins)