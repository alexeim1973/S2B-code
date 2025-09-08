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
# True if we read snapshots from the model directohry
load_data = True
#load_data = False

manual_snap_list = True
manual_snap_list = False

# Turn ON and OFF logging in functions
log = True
#log = False

# print out datasets and metadata
verbose_log = True
#verbose_log = False

# Turn ON and OFF show plotting in functions
show_plot = True
#show_plot = False

# save plots to the inage_dir folder
save_image_file = True
#save_image_file = False

# save plots as PDF
save_pdf_file = True
#save_pdf_file = False

# Plot face-on number density after rotating the dataset
verbose_bar_plot = True    
#verbose_bar_plot = False

# Plot face-on number density per age group
plot_density_face_on_per_ag = True  
#plot_density_face_on_per_ag = False

# Plot bar Amp/Phase per age group
plot_bar_amplitude = True   
plot_bar_amplitude = False

# Plot ellipticity per radius for bar fase-on, aligned to X-axis
plot_bar_ellipticity = True 
#plot_bar_ellipticity = False

# Plot bar end-on LOS sigma
plot_LOS_sigma = True
#plot_LOS_sigma = False

# Plot bar end-on LOS sigma per age group
plot_LOS_sigma_per_ag = True
#plot_LOS_sigma_per_ag = False

# Plot LOS sigma Ampplitude per Fourier moment per age group
plot_sigma_amp_per_ag_single = True 
plot_sigma_amp_per_ag_single = False

# Plot LOS sigma Amplitude with combined Fourier moment 4 and 6
plot_sigma_amp_combined = True 
plot_sigma_amp_combined = False

# Plot LOS sigma Amplitude with combined Fourier moment 4 and 6 per age group
plot_sigma_amp_per_ag_combined = True 
#plot_sigma_amp_per_ag_combined = False

"""
!!! The following timeline based parameters only make sense for a galaxy model with many snapshots.
Do not use them for TNG colections with a single snapshot pr galaxy.
"""

# Plot LOS sigma Amplitude and bar ellipticity timeline with combined Fourier moment 4 and 6 per age group
plot_sigma_amp_bar_ellip_timeline_per_ag = True
plot_sigma_amp_bar_ellip_timeline_per_ag = False

# Plot only bar ellipticity timeline per age group
plot_bar_ellip_timeline_per_ag = True
plot_bar_ellip_timeline_per_ag = False

# Plot only LOS sigma Amplitude timeline with combined Fourier moment 4 and 6 per age group
plot_sigma_amp_timeline_per_ag = True
plot_sigma_amp_timeline_per_ag = False

# Show ellipticity
show_e = True
#show_e = False

# True if the nuclear bar is considered
ncl_bar = True
ncl_bar = False

# Align the bar during the data load
bar_align = True

# If the model is a high resolution
model_high_res = True
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
    xlim, ylim = 7., 7. # kpc - primary bar plus a bit more

cmap_dens = "hot"
cmap_velo = "hot"

mass_fact = 1

grp_sw = "fixed_age" # Fixed age groups 0 < 1/3(max_age) < 2/3(max_age) < max_age
grp_sw = "eq_pop" # age groups with equal populations 1/3(total_stars)

# Sigma for Gaussian blur
blur_lst = [2,3,4,5]

barFm = 2 # Fourier moment for bar
sigmaFm = [4,6] # Fourier moment for sigma

