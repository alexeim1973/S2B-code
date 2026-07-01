__version__ = "1.1.3"
# Changelog
# 1.1.3 - Consolidated every boolean config switch from a "declare True then declare
#         False" pair (where the second line silently won) into a single assignment,
#         with a comment above each switch documenting what True and what False do.
#         No flag's effective value was changed - each switch keeps whatever value was
#         actually active in v1.1.2 before this edit.
# 1.1.2 - No functional change; version bumped to match Fourier_functions.py 1.1.2
#         (fixed the age_grp/save_file argument-binding bug, no params affected).
# 1.1.1 - No functional change; version bumped to match Fourier_functions.py 1.1.1
#         (stage 2 refactor of the sigma-shape helpers, no params affected).
# 1.1.0 - No functional change; version bumped to match Fourier_functions.py 1.1.0.
# 1.0.0 - Baseline before the code review.

# Working directories
#model = 'run732HF'
#model = 'run739HF'
model = 'TNG50' # TNG50 skinny bar from Stuart Andersson
#model = 'TNG50Ctrl' # TNG50 skinny bar from Stuart Andersson
#model = 'run741CU' # Same age, problems with breaking into age groups
#model = 'SB_models'
#model = 'SB_nogas_models' # Same age, problems with breaking into age groups
base_dir = '/home/ubuntu/projects/S2B/'
model_dir = base_dir + "models/" + model + "/"
bar_image_dir = base_dir + "images/bar-shapes/" + model + "/"
sigma_image_dir = base_dir + "images/sigma-shapes/" + model + "/"
pdf_dir = base_dir + "pdfs/" + model + "/"

# Control switches

# True: read snapshots from the model directory (normal use).
# False: reuse the already-loaded `sn` from a previous run (skips reloading).
load_data = True

# True: use the hardcoded snapshot list below instead of scanning model_dir.
# False: scan model_dir for snapshot files via mf.list_snaps().
manual_snap_list = False

# True: print progress messages from inside library functions.
# False: run library functions quietly.
log = True

# True: print detailed dataset contents and metadata for debugging.
# False: skip the extra debug output.
verbose_log = False

# True: display each plot on screen as it's produced.
# False: skip on-screen display (useful for headless/batch runs).
show_plot = True

# True: save each plot as a PNG to the relevant image directory.
# False: skip saving plot images.
save_image_file = True

# True: save the collected plots for each snapshot as a combined PDF.
# False: skip PDF export.
save_pdf_file = True

# True: plot face-on number density for the whole model (STEP 2).
# False: skip that plot.
verbose_bar_plot = True

# True: plot face-on number density per age group (STEP 4).
# False: skip that plot.
plot_density_face_on_per_ag = True

# True: plot bar amplitude/phase (Fourier moment 2) per age group (STEP 5).
# False: skip that plot.
plot_bar_amplitude = False

# True: plot bar ellipticity vs radius per age group (STEP 3).
# False: skip that plot.
plot_bar_ellipticity = True

# True: plot edge-on LOS velocity dispersion, end-on bar (STEP 7b/7c).
# False: skip those plots.
plot_LOS_sigma = True

# True: plot edge-on number density and LOS sigma per age group (STEP 8a/8b).
# False: skip those plots.
plot_LOS_sigma_per_ag = True

# True: plot sigma amplitude for a single Fourier moment per age group (STEP 8c).
# False: skip that plot.
plot_sigma_amp_per_ag_single = False

# True: plot sigma amplitude for the whole model, Fourier moments 4 and 6 combined (STEP 7d).
# False: skip that plot.
plot_sigma_amp_combined = False

# True: plot sigma amplitude per age group, Fourier moments 4 and 6 combined (STEP 8d).
# False: skip that plot.
plot_sigma_amp_per_ag_combined = False

"""
The following timeline based parameters only make sense for a galaxy model with many snapshots.
Do not use them for TNG colections with a single snapshot pr galaxy.
"""

# True: plot the sigma amplitude and bar ellipticity timeline across all snapshots.
# False: skip the timeline plot entirely.
plot_sigma_amp_bar_ellip_timeline_per_ag = False

# True: within the timeline plot, include bar ellipticity vs age.
# False: leave bar ellipticity out of the timeline plot.
plot_bar_ellip_timeline_per_ag = False

# True: within the timeline plot, include sigma amplitude vs age.
# False: leave sigma amplitude out of the timeline plot.
plot_sigma_amp_timeline_per_ag = False

# True: print the ellipticity table per age group at the end of the run.
# False: skip that printout.
show_e = True

# True: consider the nuclear bar (zoomed-in alignment/binning).
# False: consider only the primary bar.
ncl_bar = False

# True: align the bar to the x-axis during data load.
# False: leave the snapshot in its as-loaded orientation.
bar_align = True

# True: use high-resolution binning parameters (bin_width=0.1kpc, bin_arc=10deg).
# False: use the low-resolution TNG50 binning parameters (bin_width=0.2kpc, bin_arc=15deg).
model_high_res = False

if model_high_res:
    bin_width = 0.1 # kpc - for Amp/Phase calculations
    bin_arc = 10 # degrees
    if ncl_bar:
        bins = 30 # for 2D statistic calculations, both number density and sigma
        xlim, ylim = 1., 1. # kpc for nuclear bar
    else:
        bins = 50 # for a zoom-out to the radius of 6 kpc
        xlim, ylim = 6.6, 6.6 # kpc - whole model, primary bar
else:
    # TNG50 low-resolution data
    bin_width = 0.2 # kpc - for Amp/Phase calculations
    bin_arc = 15 # degrees

    bins = 30 # for 2D statistic calculations, both number density and sigma
    num_ellipses = bins/2 # half teh number of bins to match the length of ellipticity arrays from different methods
    #xlim, ylim = 5., 5. # kpc - primary bar
    #xlim, ylim = 7., 7. # kpc - primary bar plus a bit more
    xlim, ylim = 10., 10. # kpc - primary bar plus a bit more

# True: show the star count in each age-group panel title.
# False: leave star counts out of panel titles.
num_stars = True

cmap_dens = "hot"
cmap_velo = "hot"

mass_fact = 1

grp_nr = 2 # 2 (age) groups, we make the groupong dompler and we only use equal group population
grp_sw = "equal_pop" # age groups with equal populations 1/3 of total number of stars
grp_sw_title = "equal group population"

# This was used for other models like 7832HF and in Alexei M BSc thesis
#grp_nr = 3 # 3 (age) groups, this was used for other models like 7832HF and in Alexei M BSc thesis
#grp_sw = "even_age" # Even age groups 0 < 1/3(max_age) < 2/3(max_age) < max_age
#grp_sw_title = "even group age"

# Parameters for Gaussian blur
#b_alpha_lst = [1,1.5,2]
b_alpha_lst = [1]
#b_sigma_lst = [1,1.5,2,2.5,3]
b_sigma_lst = [1,2,3,4,5,6,7,8,9,10]
b_mode = "residual"
b_mode = "normalized"
#b_mode = "unsharp"

barFm = 2 # Fourier moment for bar
sigmaFm = [4,6] # Fourier moment for sigma
