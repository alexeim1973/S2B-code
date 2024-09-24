#Extract phase space data for the model
y, x, m = Data['y'], Data['x'], Data['m']

#Capture the radius of each particle and its cylindrical angle phi
R_plot = np.hypot(x, y)
phis = np.arctan2(y, x)

s2p = np.sin(2*phis)
c2p = np.cos(2*phis)

# For each angular bin calculate the bar amp and phase angle
s2p_binned = stats.binned_statistic(R_plot, m * s2p, 'sum',  bins=bins)
c2p_binned = stats.binned_statistic(R_plot, m * c2p, 'sum',  bins=bins)
mass_binned = stats.binned_statistic(R_plot, m, 'sum',  bins=bins)

s2p_sum = s2p_binned.statistic.T
c2p_sum = c2p_binned.statistic.T
mass = mass_binned.statistic.T

phi2_plot = 0.5 * np.degrees(np.arctan2(s2p_sum, c2p_sum))
a2_plot = np.hypot(s2p_sum, c2p_sum)/mass

# Find midpoint of the bins
radial_bins = s2p_binned.bin_edges[:-1] + np.diff(s2p_binned.bin_edges)/2

# We wish to locate where, after the initial settling, phi2 changes
# from constant by more than 10 degrees
# Set initial settling to be R = 1kpc and extract the first time the
# absolute value goes above 10 degrees - this is the extent of the bar
bar_ends_phi2_criterion = 10
bar_ends_phi2 = phi2_plot[(radial_bins > 1) & (abs(phi2_plot) >= 
bar_ends_phi2_criterion)][0]
bar_ends_R_phi2 = radial_bins[(radial_bins > 1) & (abs(phi2_plot) >= 
bar_ends_phi2_criterion)][0]

 # A low estimate for the bar would be half the a2 peak
 # The a2 peak is the first peak in the plot
 # Then find the half peak and its location
a2_peaks, _ = find_peaks(a2_plot) 
a2_max = a2_plot[a2_peaks[0]]

a2_max_R = radial_bins[a2_peaks[0]]
bar_ends_a2_criterion = 2

 # If this criteria is met then we do not have a bar
if len(radial_bins[(radial_bins > a2_max_R) & 
(a2_plot <= a2_max/bar_ends_a2_criterion)]) == 0:
    bar_ends_R_phi2, bar_ends_R_a2 = np.nan, np.nan
else:
    bar_ends_R_a2 = radial_bins[(radial_bins > a2_max_R) & 
    (a2_plot <= a2_max/bar_ends_a2_criterion)][0]

 # If the bar amplitude a2_max falls below 0.2 then consider the bar unformed
 # and set the radii to be nan
if a2_max < 0.2:
    bar_ends_R_phi2, bar_ends_R_a2 = np.nan, np.nan

print('For model {0}, the amp bar ends at R = {1}'.format(model, round(bar_ends_R_phi2, 2) ))
print('For model {0}, the phase bar ends at R = {1}'.format(model, round(bar_ends_R_a2, 2) ))
