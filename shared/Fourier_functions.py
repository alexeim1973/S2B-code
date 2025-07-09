# import scipy
from scipy import stats as st
import pynbody as pb
from datetime import datetime
from xml.etree.ElementInclude import include
from matplotlib.colors import LogNorm
from matplotlib.pylab import *
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Bbox
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter as gf
from scipy.signal import find_peaks as find_peaks
import os
import numpy as np
import matplotlib.pyplot as plt

from Fourier_params import *

# Reads snapshot names and param file names from the model directory
def list_snaps(model_dir):

    # list to store files
    snap_lst = []
    param_lst = []
    
    # Iterate directory
    for path in os.listdir(model_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(model_dir, path)) and '.param' in path:
            param_lst.append(path)
        elif os.path.isfile(os.path.join(model_dir, path)) and '.param' not in path:
            snap_lst.append(path)
    
    if verbose_log:
        print(snap_lst)
        print(param_lst)

    return(sorted(snap_lst),sorted(param_lst))


# Alligns the bar to x-axis, using inertia tensor
def bar_align(galaxy, nuclear = False, log = False):

    """
    Aligns the bar of pynbody galaxy simulation with the x-axis assuming the
    galaxy disc is already aligned to the XY plane using the inertial tensor.

    Function does not return any values/object. Pynbody functions effect the
    global variable which stores 'galaxy' so rotations within the functions
    are applied to input variable 'galaxy'.

    Parameters
    ----------
    galaxy : pynbody simulation object
        Galaxy object in the XY plane to be aligned.

    rbar : float
        Bar radius in simulation units e.g. kpc.

    barfrac : float
        Fraction of bar length to calculate the inertial tensor within in
        simulation units e.g. kpc.

    zlim : float
        Vertical limit to calculate intertial tensor within in simulation units
        e.g. kpc. Useful in galaxies with thick discs and weak bars.

    log : Bool
        Flag to output print statements.

    Returns
    -------
    None

    """

    if log:
            print('* Aligning the data...')

    if nuclear:
            # rbar = 1, barfrac = 0.5, zlim = 0.5 - zooming in to the central area 1 by 1 kpc
            # rbar = 1, barfrac = 0.5, zlim = 0.25 - adjusting vertical limit to 0.25 kpc
            # rbar = 1, barfrac = 0.75, zlim = 0.25 - increasing bar fraction to 75%
            rbar = 1.0
            barfrac = 0.75
            zlim = 0.25
            #bar_align(sim,rbar = 1.0, barfrac = 0.75, zlim = 0.25, log = log)
    else:
            # Primary bar alignment # rbar = 3, barfrac = 0.5, zlim = 0.5
            rbar = 3.0
            barfrac = 0.5
            zlim = 0.5
            #bar_align(sim,rbar = 3.0 ,barfrac = 0.5 ,zlim = 0.5, log = log)
            # Nuclear bar alignment?
            #bar_align(sim,rbar = 3.0 ,barfrac = 1.0 ,zlim = 0.5, log = log)

    if np.isnan(rbar):
        if log:
            print('* Bar undefined, using 1 kpc')
        rbar = 1.0
    elif rbar*barfrac < 1.:
        rbar = 1
        if log:
            print('* Short Bar, using 1 kpc')
    else:
        rbar = rbar*barfrac
        if log:
            print('* Bar defined, aligning to {} kpc'.format(rbar))

    if log:
        print('* Realigning bar using |z| < {} '.format(zlim))

    zfilt = pb.filt.LowPass('z',zlim)&pb.filt.HighPass('z',-zlim)
    rfilt = pb.filt.LowPass('rxy',rbar)

    x = np.array(galaxy[zfilt&rfilt].star['pos'].in_units('kpc'))[:,0]
    y = np.array(galaxy[zfilt&rfilt].star['pos'].in_units('kpc'))[:,1]
    m = np.array(galaxy.star[zfilt&rfilt]['mass'])

    #Calculate the inertia tensor
    I_yy, I_xx, I_xy = np.sum(m*y**2),np.sum(m*x**2),np.sum(m*x*y)
    I = np.array([[I_yy, -I_xy], [-I_xy, I_xx]])

    #Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(I)
    lowest = eigenvalues.argmin()
    maj_axis = eigenvectors[:, lowest]

    #Get the angle we need to rotate by
    r_angle = np.degrees(np.arctan2(maj_axis[1], maj_axis[0]))

    galaxy.rotate_z(-r_angle)

    if log:
        print('* Bar realigned by {0:3.2f} degrees'.format(r_angle))

    return None


# Loads snapshot in the memory using Pynbody
def pbload(filename,  paramname=None, nuclear=False, align=False, log=False):
    
    if verbose_log:
        print("From function pbload of module my_functions.py.")
        print("Data file:", filename)
        print("Param file:", paramname)
        print("Nuclear bar:", nuclear)
        print("Align bar to the X-acis:", align)
        print("Verbose logging:", log)

    # Loading data from the file
    if log:
        print('* Loading data from file', filename, '...')
    
    if '::' in filename:
        filename, species = filename.split('::')
        sim = pb.load(filename, paramname=paramname)
        sim = getattr(sim, species)
    else:
        sim = pb.load(filename, paramname=paramname)

    # Centering the data
    if log:
        print('* Centering the data...')

    #Centre the stars using a hybrid method - potential minimum and shrinking sphere
    pb.analysis.halo.center(sim.s, mode='hyb')
    
    #Rotate the simulation so that we see the stars face on
    pb.analysis.angmom.faceon(sim.s)

    # Aligning the data
    if align:
        bar_align(sim, nuclear, log)

    return sim


# Extracts component of NumPy array
def extract_np(sim):
    x, y, z = np.array(sim.s['x']), np.array(sim.s['y']), np.array(sim.s['z'])
    m = np.array(sim.s['mass'])
    age = np.array(sim.s['age'])
    tf = np.array(sim.s['tform'])

    return x,y,z,m,age,tf


# Masks component of NumPy array within xlim by ylim slab
def mask_np(x,y,z,m,age,tf,xlim,ylim):
    mask = (abs(x) < xlim) & (abs(y) < ylim)
    x, y, z = x[mask], y[mask], z[mask]
    m = m[mask]
    age, tf = age[mask], tf[mask]

    return x,y,z,m,age,tf


# This function extracts a sub-list of limited dimentions m by m from original list n by n, centered as nested squares.
def extract_sublist(original_list, m):
    n = len(original_list)
    start_row = (n - 2*m) // 2
    end_row = start_row + 2*m
    start_col = (n - 2*m) // 2
    end_col = start_col + 2*m

    sub_list = []
    for i in range(start_row, end_row):
        sub_list.append(original_list[i][start_col:end_col])

    return sub_list


# This function extracts a full sub-set r by r from original stat2d datacet, centered as nested squares.
def extract_subset(data, radius, center=None):
    bins = data.shape[0]
    if center is None:
        center = bins // 2
    return data[center - radius:center + radius, center - radius:center + radius]


# This function calculates sums on rows and collumns for a given 2D list.
def sum_columns_and_rows(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(num_rows)) for j in range(num_cols)]

    return row_sums, col_sums


# Divide star population in 3 groups 
def grouping(sim):
    if grp_sw == "fixed_age": # with fixed age dividers 0 < 1/3 < 2/3 < 1
        min_age = np.min(sim.star['age'])
        max_age = np.max(sim.star['age'])
        edges = np.linspace(min_age, max_age, 4)  # Divides into 3 equal age bins
        groups = []
        for i in range(3):
            if i < 2:
                mask = (sim.star['age'] >= edges[i]) & (sim.star['age'] < edges[i+1])
            else:
                mask = (sim.star['age'] >= edges[i]) & (sim.star['age'] <= edges[i+1])  # include max in last bin
            grp = sim.star[mask]
            groups.append(grp)
    elif grp_sw == "eq_pop": # by age with equal population 1/3(total_stars) per group
        # Sort stars by age
        sorted_indices = np.argsort(sim.star['age'])
        sorted_stars = sim.star[sorted_indices]

        # Split into three equal groups
        total_stars = len(sorted_stars)
        group_size = total_stars // 3

        groups = [
            sorted_stars[0:group_size],                            # Youngest third
            sorted_stars[group_size:2*group_size],                 # Intermediate third
            sorted_stars[2*group_size:],                           # Oldest third (may be slightly larger if not divisible by 3)
        ]

    return groups


# Pring stellar ages for snmapshot
def print_ages(sim,entity):
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    print(f'** Stars in {entity} - ', len(sim.star))
    print('**** Min stellar age - ' + str(min_age) + ' Gyr.')
    print('**** Max stellar age - ' + str(max_age) + ' Gyr.')


# Check if snapshot min and max ages are different
def check_ages(sim):
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit # Stop if min and max ages are the same


# Plots face-on number dencity
def plot_density(cmap,sim,snap,image_dir,save_file=True):

    # Print snapshot max and min ages and number of stars
    if verbose_log:
        print_ages(sim,"snapshot")

    # Extract NP arrays and mask central area xlim by ylim
    x,y,z,m,age,tf = extract_np(sim)
    x,y,z,m,age,tf = mask_np(x,y,z,m,age,tf,xlim,ylim)

    # Number density statistics face-on for stellar population by age group
    df_stat2d,df_xedges,df_yedges,df_binnum2d = st.binned_statistic_2d(x, y, z,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)

    if show_e:
        unit = 2*xlim/bins
        if verbose_log:
            print("Unit in kpc:", round(unit,2))
        e = ellipticity(unit,bins,df_stat2d)
        e1 = ellipticity_from_moments(df_stat2d, xlim, bins)
        print('*** Ellipticity:', e)
        print('*** Ellipticity using moments:', e1)

    x_panels = 1
    y_panels = 1

    figsize_x = 5*x_panels    # inches
    figsize_y = 5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))

    image = plt.imshow(df_stat2d.T, 
                origin = 'lower',
                extent = [-xlim, xlim, -ylim, ylim ],
                norm = LogNorm(),
                cmap = cmap)
    xcent = (df_xedges[1:] + df_xedges[:-1]) / 2
    ycent = (df_yedges[1:] + df_yedges[:-1]) / 2
    plt.contour(xcent, ycent, np.log10(df_stat2d.T), linewidths = 0.5, linestyles = 'dashed', colors = 'k')
    if show_e:
        plt.title(snap + " - face-on num density. Bar ellipticity " + str(e1))
    else:
        plt.title(snap + " - face-on num density.")
    axes = plt.gca()
    axes.set_xlabel('x [kpc]')
    axes.set_ylabel('y [kpc]')

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(image, cax=cax, orientation='vertical')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_density_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    return fig


# Plots number dencity for age groups
def plot_density_by_age(cmap,sim,pos,xlim,ylim,bins,snap,image_dir,e_list=False,save_file=True,show_plot=True,verbose_log=False):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit

    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
            grp_min_age = round(min(grp.star['age']),2)
            grp_max_age = round(max(grp.star['age']),2)
            print('***** Min stellar age in group - ' + str(grp_min_age) + ' Gyr.')
            print('***** Max stellar age in group - ' + str(grp_max_age) + ' Gyr.')
            if e_list:
                print("*** Bar ellipticity - ", e_list[age_grp-1])
        
        #Extract phase space data for the model for stars in the group
        z_, x_, y_ = grp.star['z'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        dfg_stat2d,dfg_xedges,dfg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, z_,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(dfg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                image = axes[i].imshow(stat2d_lst[i], 
                            origin = 'lower',
                            extent = [-xlim, xlim, -ylim, ylim ],
                            norm = LogNorm(),
                            cmap = cmap)
                xcent = (dfg_xedges[1:] + dfg_xedges[:-1]) / 2
                ycent = (dfg_yedges[1:] + dfg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, np.log10(stat2d_lst[i]), linewidths = 0.5, linestyles = 'dashed', colors = 'k')
                if e_list:
                    axes[i].title.set_text("Age group " + str(i+1) + " e = " + str(e_list[i]))
                else:
                    axes[i].title.set_text("Age group " + str(i+1))
                circle1 = plt.Circle((0, 0), 0.5, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 0.75, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " num density.")
    plt.setp(axes[:], xlabel = 'x [kpc]')
    plt.setp(axes[0], ylabel = 'y [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + pos + '_density_by_age_3grp_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots number dencity for age groups using group switch
def plot_density_by_age_grp_sw(cmap,sim,pos,snap,image_dir,e_list=False,save_file=True):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")
        #print("E-list:", e_list)
    
    # Check if min age is not equal to max age    
    check_ages(sim)

    # Number density statistics per age group
    age_grp = 0

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        if verbose_log:
            print_ages(grp,"group")
            if e_list:
                print("***** Bar ellipticity - ", e_list[age_grp-1])
        
        #Extract phase space data for the model for stars in the group
        z_, x_, y_ = grp.star['z'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        dfg_stat2d,dfg_xedges,dfg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, z_,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(dfg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                image = axes[i].imshow(stat2d_lst[i], 
                            origin = 'lower',
                            extent = [-xlim, xlim, -ylim, ylim ],
                            norm = LogNorm(),
                            cmap = cmap)
                xcent = (dfg_xedges[1:] + dfg_xedges[:-1]) / 2
                ycent = (dfg_yedges[1:] + dfg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, np.log10(stat2d_lst[i]), linewidths = 0.5, linestyles = 'dashed', colors = 'k')
                if e_list:
                    axes[i].title.set_text("Age group " + str(i+1) + " e = " + str(e_list[i]))
                else:
                    axes[i].title.set_text("Age group " + str(i+1))
                circle1 = plt.Circle((0, 0), 0.5, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 0.75, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    suptit_name = snap.replace(".gz","") + " num density " + grp_sw 
    fig.suptitle(suptit_name)
    plt.setp(axes[:], xlabel = 'x [kpc]')
    plt.setp(axes[0], ylabel = 'y [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + pos + '_dens_by_age_3grp_' + str(xlim) + 'kpc_' + grp_sw + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# This function calculates a bar ellipticity
def ellipticity(unit,bins,list):
    q_x = 0
    q_y = 0

    for R in range (1, int(bins/2)+1):
        #print("Radius:", R)

        sub_list = extract_sublist(list, R)
        #print(sub_list)
        #for e in sub_list:
        #    print(e)

        row_sums, col_sums = sum_columns_and_rows(sub_list)
        #print("Row sums:", row_sums)
        #print("Column sums:", col_sums)

        r = unit*R

        q_y = q_y + row_sums[R]*(r**2)
        q_x = q_x + col_sums[R]*(r**2)
        #print("Qx:", round(q_x,2))
        #print("Qy:", round(q_y,2))

    e = 1 - sqrt(q_y / q_x)

    return(round(e,2))


# This function calculates a bar ellipticity usign moments
def ellipticity_from_moments(density, xlim, bins):
    # Create grid coordinates
    x = np.linspace(-xlim, xlim, bins)
    y = np.linspace(-xlim, xlim, bins)
    X, Y = np.meshgrid(x, y)

    # Flatten arrays
    x_flat = X.flatten()
    y_flat = Y.flatten()
    density_flat = density.flatten()

    # Compute weighted moments
    x_mean = np.average(x_flat, weights=density_flat)
    y_mean = np.average(y_flat, weights=density_flat)

    # === DIAGNOSTIC: print COM shift ===
    if verbose_log:
        com_shift = np.sqrt(x_mean**2 + y_mean**2)
        print(f"[DIAG] COM shift: x = {x_mean:.2f} kpc, y = {y_mean:.2f} kpc, |COM| = {com_shift:.2f} kpc")

    x_dev = x_flat - x_mean
    y_dev = y_flat - y_mean

    Ixx = np.average(x_dev**2, weights=density_flat)
    Iyy = np.average(y_dev**2, weights=density_flat)
    Ixy = np.average(x_dev * y_dev, weights=density_flat)

    # Calculate eigenvalues
    I = np.array([[Ixx, Ixy],
                  [Ixy, Iyy]])
    eigvals = np.linalg.eigvalsh(I)
    a2, b2 = np.sort(eigvals)[::-1]  # a2 ≥ b2

    # Axis ratio
    axis_ratio = np.sqrt(b2 / a2)
    ellipticity = 1 - axis_ratio

    return round(ellipticity, 2)


# Plots bar eööipticity
def bar_ellipticity_by_age(sim,xlim,ylim,bins,snap,image_dir,save_file=True,show_plot=True,verbose_log=True):

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    # List of ellipticities and radii per age group for plotting
    e_list = [[],[],[]]
    r_list = [[],[],[]]
    e_age_grp = []

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))

        #Extract phase space data for the model for stars in the group
        z_, x_, y_ = grp.star['z'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population in the group
        dfg_stat2d,dfg_xedges,dfg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, z_,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)

        unit = 2*xlim/bins
        if verbose_log:
            print("Unit in kpc:", round(unit,2))

        e = ellipticity(unit,bins,dfg_stat2d)
        print('*** Age group', age_grp, 'ellipticity:', e)
        e_age_grp.append(e)

        if verbose_log:
            for radius in range(1,int(bins/2)+1):
                sub_list = extract_sublist(dfg_stat2d, radius)
                e = ellipticity(unit,2*radius,sub_list)
                e_list[age_grp-1].append(e)
                r_list[age_grp-1].append(round(radius*unit,2))
                if verbose_log:
                    print('*** Radius:', round(radius*unit,2), " kpc, ", 'ellipticity:', e)

        x_panels = 3
        y_panels = 1
        figsize_x = 3*x_panels      # inches
        figsize_y = 3.5*y_panels    # inches

        # Make the figure and sub plots
        fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
        for i in range(x_panels):
            image = axes[i].plot(r_list[i],e_list[i])
            axes[i].title.set_text("Age group " + str(i+1))
            # axes[i].set_ylim(min(e_list[age_grp-1]), max(e_list[age_grp-1]))

            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            #cbar = fig.colorbar(image, cax=cax, orientation='vertical')
            #cbar.set_label(cbar_label_lst[i])
            if i > 0:
                axes[i].set_yticklabels([])

            fig.tight_layout()
            fig.suptitle(snap.replace(".gz","") + " bar ellipticity per age group.")
            plt.setp(axes[:], xlabel = 'R [kpc]')
            plt.setp(axes[0], ylabel = 'e')

            if save_file:
                image_name = image_dir + snap.replace(".gz","") + '_ellipticity_by_age_3grp_' + str(xlim) + 'kpc' + '.png'
                plt.savefig(image_name)
                print("Image saved to",image_name)
            else:
                print("Image saving is turned off.")
            if show_plot:
                plt.show()
            else:
                print("On-screen ploting is turned off.")

    return e_age_grp


from scipy.interpolate import interp1d

def ellipticity_from_ellipse_fit_interp2(density, unit, r_max=None, bins=None):
    """
    Calculate bar ellipticity from ellipse fitting and interpolate onto fixed radial grid.

    Parameters:
        density (2D array): 2D number density map.
        unit (float): Size of one bin in physical units (e.g., kpc).
        r_max (float): Max radius for interpolation (e.g., xlim). Defaults to half image width.
        bins (int): Number of bins in density map (optional, used if r_max is not given).
        verbose (bool): Print diagnostics.
    
    Returns:
        r_interp (list): Radii in physical units [kpc].
        e_interp (list): Interpolated ellipticities.
    """
    from photutils.isophote import EllipseGeometry, Ellipse
    from scipy.interpolate import interp1d

    ny, nx = density.shape
    x0 = nx // 2
    y0 = ny // 2

    geometry = EllipseGeometry(x0, y0, sma=5, eps=0.2, pa=0)
    ellipse = Ellipse(density, geometry)

    isolist = ellipse.fit_image()

    if verbose_log:
        print(f"[DIAG] Ellipse fitting returned {len(isolist)} successful isophotes")

    # Extract radii and ellipticities from isophotes
    sma_list = np.array([iso.sma for iso in isolist])  # in pixels
    e_list = np.array([1 - iso.eps for iso in isolist])  # eps = 1 - b/a => ellipticity = eps
    r_list = sma_list * unit  # Convert to kpc

    # Return early if no usable data
    if len(r_list) == 0 or len(e_list) == 0:
        print("[WARNING] Ellipse fitting returned empty arrays.")
        return [], []

    # Set interpolation radius range
    if r_max is None:
        r_max = r_list[-1]
    elif bins is not None:
        r_max = min(r_max, bins * unit / 2)

    # Define fixed radial spacing (like in my other methods)
    r_interp = np.arange(unit, r_max + unit, unit)

    # Interpolate ellipticity using linear interpolation
    interp_func = interp1d(r_list, e_list, kind='linear', bounds_error=False, fill_value="extrapolate")
    e_interp = interp_func(r_interp)

    return r_interp.tolist(), e_interp.tolist()


def ellipticity_from_ellipse_fit_interp(density, unit, r_max=None, bins=None):
    """
    Calculate bar ellipticity from ellipse fitting and interpolate onto fixed radial grid.

    Parameters:
        density (2D array): 2D number density map.
        unit (float): Size of one bin in physical units (e.g., kpc).
        r_max (float): Max radius for interpolation (e.g., xlim).
        bins (int): Number of bins in density map (used if r_max not given).

    Returns:
        r_interp (list): Radii in physical units [kpc].
        e_interp (list): Interpolated ellipticities.
    """
    from photutils.isophote import EllipseGeometry, Ellipse
    from scipy.interpolate import interp1d
    import numpy as np

    ny, nx = density.shape
    x0 = nx // 2
    y0 = ny // 2

    geometry = EllipseGeometry(x0, y0, sma=1, eps=0.2, pa=0)
    ellipse = Ellipse(density, geometry)
    isolist_raw = ellipse.fit_image()

    # Filter only valid isophotes
    isolist = [iso for iso in isolist_raw if iso.valid]

    if verbose_log:
        print(f"[DIAG] Ellipse fitting returned {len(isolist)} valid isophotes")

    # Return early if no usable data
    if len(isolist) < 2:
        print("[WARNING] Not enough valid isophotes for interpolation.")
        return [], []

    # Extract radius and ellipticity
    sma_list = np.array([iso.sma for iso in isolist])         # in pixels
    e_list = np.array([1 - iso.eps for iso in isolist])       # eps = 1 - b/a
    r_list = sma_list * unit                                  # convert to kpc

    # Set interpolation radius range
    if r_max is None:
        r_max = r_list[-1]
    elif bins is not None:
        r_max = min(r_max, bins * unit / 2)

    r_interp = np.arange(unit, r_max + unit, unit)

    # Interpolate ellipticity using linear interpolation
    interp_func = interp1d(r_list, e_list, kind='linear', bounds_error=False, fill_value="extrapolate")
    e_interp = interp_func(r_interp)
    
    if verbose_log:
        print(f"[DIAG] First 5 interpolated e: {e_interp[:5]}")

    return r_interp.tolist(), e_interp.tolist()




from photutils.isophote import Ellipse, EllipseGeometry

def ellipticity_from_ellipse_fit(density, unit):
    center = density.shape[0] // 2
    geometry = EllipseGeometry(x0=center, y0=center, sma=5, eps=0.1, pa=0)
    ellipse = Ellipse(density, geometry)  # ← pass raw array directly
    isolist = ellipse.fit_image()

    if isolist is None or len(isolist) == 0:
        if verbose_log:
            print("Isolist is None!")
        return [], []

    radii = [iso.sma * unit for iso in isolist]
    ellipticities = [iso.eps for iso in isolist]

    return radii, ellipticities


# Plots bar eööipticity using grouping switch
def bar_ellipticity_by_age_grp_sw(sim,snap,image_dir,save_file=True):

    x_panels = 3
    y_panels = 1
    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")
    
    # Check if min age is not equal to max age    
    check_ages(sim)
            
    # Number density statistics per age group
    age_grp = 0

    # List of ellipticities and radii per age group for plotting
    e1_list = [[],[],[]]
    e2_list = [[],[],[]]
    r1_list = [[],[],[]]
    r2_list = [[],[],[]]
    e3_list = [[],[],[]]
    r3_list = [[],[],[]]
    e_age_grp = []

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        i = age_grp - 1
        if verbose_log:
            print_ages(grp,"group")

        #Extract phase space data for the model for stars in the group
        z_, x_, y_ = grp.star['z'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        dfg_stat2d,dfg_xedges,dfg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, z_,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)

        unit = 2*xlim/bins
        if verbose_log:
            print("Unit in kpc:", round(unit,2))

        e1 = ellipticity(unit,bins,dfg_stat2d)
        e2 = ellipticity_from_moments(dfg_stat2d, xlim, bins)

        # Ellipse fitting
        #r3, e3 = ellipticity_from_ellipse_fit(dfg_stat2d, unit)
        r3, e3 = ellipticity_from_ellipse_fit_interp(dfg_stat2d, unit, xlim, bins)
        if r3 and e3:
            r3_list[i] = r3
            e3_list[i] = e3
            print('*** Age group', age_grp, 'max ellipticity using fitting:', max(e3))
        else:
            print(f"[WARNING] Skipping ellipse fitting plot for age group {age_grp}")

        print('*** Age group', age_grp, 'ellipticity:', e1)
        print('*** Age group', age_grp, 'ellipticity using monments:', e2)
        if(e3): print('*** Age group', age_grp, 'max ellipticity using fitting:', max(e3))
        # Pick the ellipticity method for the rest of calculations
        e_age_grp.append(e2)

        for radius in range(1,int(bins/2)+1):
            sub_list = extract_sublist(dfg_stat2d, radius)
            e1 = ellipticity(unit,2*radius,sub_list)
            e1_list[i].append(e1)
            r1_list[i].append(round(radius*unit,2))

            subset = extract_subset(dfg_stat2d, radius)
            subset_bins = subset.shape[0]
            subset_xlim = unit * radius

            e2 = ellipticity_from_moments(subset, subset_xlim, subset_bins)
            e2_list[i].append(e2)
            r2_list[i].append(round(subset_xlim,2))
                
            if verbose_log:
                print('*** Radius:', round(radius*unit,2), " kpc, ", 'ellipticity:', e1)
                print('*** Radius:', round(subset_xlim,2), " kpc, ", 'ellipticity from moments:', e2)
                print('*** Radius:', round(subset_xlim,2), " kpc, ", 'ellipticity from fitting:', e3)

        
        image = axes[i].plot(r1_list[i],e1_list[i], color = "blue", label="Quadrupole method")
        image = axes[i].plot(r2_list[i],e2_list[i], color = "red", label="Inertia tensor method")
        axes[i].plot(r3_list[i], e3_list[i], color="green", label="Ellipse fitting method")
        axes[i].legend()
        axes[i].title.set_text("Age group " + str(i+1))

    # Add legend to whole figure
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)

    # Layout tweaks
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    suptit_name = snap.replace(".gz","") + " bar ellipticity per age group " + grp_sw 
    fig.suptitle(suptit_name)            
    plt.setp(axes[:], xlabel = 'R [kpc]')
    plt.setp(axes[0], ylabel = 'e')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_ellipticity_by_age_3grp_' + str(xlim) + 'kpc_' + grp_sw + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    return e_age_grp


# Plots bar length estimation using amplitude and phase of Fourier moment 2
def bar_length(sim,bin_width,xlim,snap,image_dir,save_file=True):
    #Extract phase space data for the model
    x, y, m = sim.s['x'], sim.s['y'], sim.s['mass']

    #Capture the radius of each particle and its cylindrical angle phi
    R_plot = np.hypot(x, y)
    phis = np.arctan2(y, x)

    bins = int(R_plot.max()/bin_width)

    s2p = np.sin(2*phis)
    c2p = np.cos(2*phis)

    # For each angular bin calculate the bar amp and phase angle
    s2p_binned = st.binned_statistic(R_plot, m * s2p, 'sum',  bins=bins)
    c2p_binned = st.binned_statistic(R_plot, m * c2p, 'sum',  bins=bins)
    mass_binned = st.binned_statistic(R_plot, m, 'sum',  bins=bins)

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
    bar_ends_phi2 = phi2_plot[(radial_bins > 1) & (abs(phi2_plot) >= bar_ends_phi2_criterion)][0]
    bar_ends_R_phi2 = radial_bins[(radial_bins > 1) & (abs(phi2_plot) >= bar_ends_phi2_criterion)][0]

    # A low estimate for the bar would be half the a2 peak
    # The a2 peak is the first peak in the plot
    # Then find the half peak and its location
    a2_peaks, _ = find_peaks(a2_plot) 
    a2_max = a2_plot[a2_peaks[0]]

    a2_max_R = radial_bins[a2_peaks[0]]
    bar_ends_a2_criterion = 2

    if len(radial_bins[(radial_bins > a2_max_R) & (a2_plot <= a2_max/bar_ends_a2_criterion)]) == 0:
        # If this criteria is met then we do not have a bar
        print('The bar is not present.')
        bar_ends_R_phi2, bar_ends_R_a2 = np.nan, np.nan
    elif a2_max < 0.2:
        # If the bar amplitude a2_max falls below 0.2 then consider the bar unformed and set the radii to be nan
        print('The bar amplitude is', round(a2_max,2), '- below 0.2.')
        bar_ends_R_phi2, bar_ends_R_a2 = np.nan, np.nan
    else:
        bar_ends_R_a2 = radial_bins[(radial_bins > a2_max_R) & (a2_plot <= a2_max/bar_ends_a2_criterion)][0]

    print('For model {0}, the amp bar ends at R = {1} kpc'.format(snap, round(bar_ends_R_phi2, 2) ))
    print('For model {0}, the phase bar ends at R = {1} kpc'.format(snap, round(bar_ends_R_a2, 2) ))

    #Plot the amplitude diagram first
    xlab = r'$R \rm \enspace [kpc]$'
    ylab = r'$A_2(R) [kpc]$'
    y2lab = r'$\phi_2(R) \enspace \rm [deg.]$'

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    ax = axes
    ax2 = ax.twinx()
    fs = 16

    ax.plot(radial_bins, a2_plot, c='r', label='r$A_2(R)$')
    ax2.plot(radial_bins, phi2_plot, c='b', label='r$\phi_2(R)$')
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax2.tick_params(axis='both', which='both', labelsize=fs)
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs, c='r')
    ax2.set_ylabel(y2lab, fontsize=fs, c='b')
    ax.axvline(bar_ends_R_a2, c='r', ls='--')
    ax.axvline(bar_ends_R_phi2, c='b', ls='--')
    ax.set_xlim(0., xlim)

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_bar_amp_phase_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    return None


# Plots bar length estimation for age group using amplitude and phase of Fourier moment 2
def bar_length_Fm(sim,Fm,snap,image_dir,age_grp=0,save_file=True):
    
    if verbose_log:
        # Debug output
        print("Snapshot xlim from the parameter file:", xlim)

    bar_xlim = xlim + 0.5

    #Extract phase space data for the model
    x, y, m = sim.s['x'], sim.s['y'], sim.s['mass']

    #Calculate the radius of each particle and its cylindrical angle phi
    R_plot = np.hypot(x, y) # Hypotenuse of right angle triangle with x,z
    phis = np.arctan2(y, x)
    
    # Calculate bins from a pre-defined bin width in kpc
    bins = int(R_plot.max()/bin_width)
    print("Bin width: ", bin_width, " kpc -> number of bins:", bins)

    Fm = Fm # Fourier component 4
    sFp = np.sin(Fm * phis) # for each mass particle = star
    cFp = np.cos(Fm * phis) # for each mass particle = star
    
    if verbose_log:
        # Debug output
        print("Bar Fourier moment:", Fm)
        print("Size of sFp array", len(sFp))
        print("Size of cFp array", len(cFp))

    # For each angular bin calculate the bar amp and phase angle
    sFp_binned = st.binned_statistic(R_plot, m * sFp, 'sum',  bins=bins)
    cFp_binned = st.binned_statistic(R_plot, m * cFp, 'sum',  bins=bins)
    mass_binned = st.binned_statistic(R_plot, m, 'sum',  bins=bins)

    sFp_sum = sFp_binned.statistic.T
    cFp_sum = cFp_binned.statistic.T
    mass = mass_binned.statistic.T

    if verbose_log:
        # Debug output
        print("Size of sFp_sum array", len(sFp_sum))
        print("Size of cFp_sum array", len(cFp_sum))
        print("Size of mass_sum array", len(mass))
        print("Sum of mass_sum array", np.nansum(mass))
        print("Min value of mass_sum array", np.nanmin(mass))
        print("Max value of mass_sum array", np.nanmax(mass))
        print("All values of mass_sum array", mass)

    phiF_plot = (1/Fm) * np.degrees(np.arctan2(sFp_sum, cFp_sum))
    aF_plot = np.hypot(sFp_sum, cFp_sum)/mass

    # Find midpoint of the bins
    radial_bins = sFp_binned.bin_edges[:-1] + np.diff(sFp_binned.bin_edges)/2
    
    # We wish to locate where, after the initial settling, phi4 changes
    # from constant by more than 10 degrees
    # Set initial settling to be R = 1kpc and extract the first time the
    # absolute value goes above 10 degrees - this is the extent of the bar
    bar_ends_phiF_criterion = 10
    bar_ends_phiF = phiF_plot[(radial_bins > 1) & (abs(phiF_plot) >=
                               bar_ends_phiF_criterion)][0]
    bar_ends_R_phiF = radial_bins[(radial_bins > 1) & (abs(phiF_plot) >=
                              bar_ends_phiF_criterion)][0]
    
    # A low estimate for the bar would be half the a2 peak
    # The a2 peak is the first peak in the plot
    # Then find the half peak and its location
    aF_peaks, _ = find_peaks(aF_plot)
    aF_max = aF_plot[aF_peaks[0]]
    
    aF_max_R = radial_bins[aF_peaks[0]]
    bar_ends_aF_criterion = 2
    
    # If this criteria is met then we do not have a bar
    if len(radial_bins[(radial_bins > aF_max_R) & (aF_plot <= aF_max/bar_ends_aF_criterion)]) == 0:
        bar_ends_R_phiF, bar_ends_R_aF = np.nan, np.nan
        # If the bar amplitude aF_max falls below 0.2 then consider the bar unformed
        # and set the radii to be nan
    elif aF_max < 0.2:
        bar_ends_R_phiF, bar_ends_R_aF = np.nan, np.nan
    else:
        bar_ends_R_aF = radial_bins[(radial_bins > aF_max_R) &
                            (aF_plot <= aF_max/bar_ends_aF_criterion)][0]

    print('For model {0}, the PHASE bar ends at R = {1}'.format(snap, round(bar_ends_R_phiF, 2) ))
    print('For model {0}, the AMPLITUDE bar ends at R = {1}'.format(snap, round(bar_ends_R_aF, 2) ))

    # Plot the amplitude diagram
    xlab = r'$R \rm \enspace [kpc]$'
    ylab = r'$A_2(R)$'
    y2lab = r'$\phi_2(R) \enspace \rm [deg.]$'

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    ax = axes
    ax2 = ax.twinx()
    fs = 8

    ax.plot(radial_bins, aF_plot, c='r', label='r$A_2(R)$')
    ax2.plot(radial_bins, phiF_plot, c='b', label='r$\phi_2(R)$')
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax2.tick_params(axis='both', which='both', labelsize=fs)
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs, c='r')
    ax2.set_ylabel(y2lab, fontsize=fs, c='b')
    # We do not plot the analysis conditions for bars from Stuart code for sigma right now.
    ax.axvline(bar_ends_R_aF, c='r', ls='--')
    ax.axvline(bar_ends_R_phiF, c='b', ls='--')
    ax.set_xlim(0., bar_xlim)

    title = snap.replace(".gz","") + "Amp-Phase Fm " + str(Fm)
    if age_grp != 0:
        title = title + " age group " + str(age_grp) 
    ax.title.set_text(title)

    age_group = ""
    if age_grp != 0:
        age_group = f"age_grp_{str(age_grp)}_{grp_sw}_"

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_bar_length_' + age_group + str(xlim) + 'kpc.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    return None


# Wrap function for age groups, calls bar_length_Fm
def bar_length_by_age_Fm(sim,Fm,snap,image_dir,save_file=True):

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
    
        bar_length_Fm(grp,Fm,snap,image_dir,age_grp,save_file)

    return None


# Wrap function for age groups using grouping switch, calls bar_length_Fm
def bar_length_by_age_grp_sw_Fm(sim,Fm,snap,image_dir,save_file=True):

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")
    
    # Check if min age is not equal to max age    
    check_ages(sim)
            
    # Bar amplitude per age group
    age_grp = 0

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        if verbose_log:
            print_ages(grp,"group")
    
        bar_length_Fm(grp,Fm,snap,image_dir,age_grp,save_file)

    return None


# Plots edge-on sigma
def plot_sigma(cmap,sim,xlim,ylim,bins,snap,image_dir,save_file=True,show_plot=True,verbose_log=False):

    #Extract phase space data for the model
    vz_, x_, y_ = sim.star['vz'], sim.star['x'], sim.star['y']
    
    # Number density statistics face-on for stellar population by age group
    vd_stat2d,vd_xedges,vd_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
    
    x_panels = 1
    y_panels = 1

    figsize_x = 5*x_panels    # inches
    figsize_y = 5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    image = plt.imshow(vd_stat2d, 
                origin = 'lower',
                extent = [-xlim, xlim, -ylim, ylim ],
                cmap = cmap)
    xcent = (vd_xedges[1:] + vd_xedges[:-1]) / 2
    ycent = (vd_yedges[1:] + vd_yedges[:-1]) / 2
    plt.contour(xcent, ycent, vd_stat2d, linewidths = 2, linestyles = 'dashed', colors = 'w')
    circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
    circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
    axes = plt.gca()
    axes.add_patch(circle1)
    axes.add_patch(circle2)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(image, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " LOS velocity dispersion.")
    axes.set_xlabel('y [kpc]')
    axes.set_ylabel('z [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_sigma_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del vd_stat2d

    return fig


# Plots edge-on sigma using the unsharp mask
def plot_sigma_umask(cmap,sim,xlim,ylim,bins,blur,snap,image_dir,save_file=True,show_plot=True,verbose_log=False):

    # Extract phase space data for the model
    vz_, x_, y_ = sim.star['vz'], sim.star['x'], sim.star['y']

    # Number density statistics face-on for stellar population by age group
    vd_stat2d, vd_xedges, vd_yedges, df_binnum2d = st.binned_statistic_2d(
        x_, y_, vz_,
        statistic='std',
        range=[[-xlim, xlim], [-ylim, ylim]],
        bins=bins
    )

    # Apply Unsharp Mask
    blurred = gf(vd_stat2d, sigma=blur)  # Apply Gaussian blur
    sharpened = vd_stat2d + (vd_stat2d - blurred)  # Unsharp masking

    x_panels = 1
    y_panels = 1

    figsize_x = 5 * x_panels  # inches
    figsize_y = 5 * y_panels  # inches

    # Make the figure and sub plots
    fig, axes = plt.subplots(y_panels, x_panels, figsize=(figsize_x, figsize_y))

    # Plot the sharpened image
    img = axes.imshow(sharpened,
                      origin='lower',
                      extent=[-xlim, xlim, -ylim, ylim],
                      cmap=cmap)

    # Contours
    xcent = (vd_xedges[1:] + vd_xedges[:-1]) / 2
    ycent = (vd_yedges[1:] + vd_yedges[:-1]) / 2
    axes.contour(xcent, ycent, sharpened, linewidths=2, linestyles='dashed', colors='w')

    # Adding circles
    circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
    circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
    axes.add_patch(circle1)
    axes.add_patch(circle2)

    # Colorbar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(img, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz", "") + " LOS velocity dispersion (sharp " + str(blur) + ").")
    axes.set_xlabel('y [kpc]')
    axes.set_ylabel('z [kpc]')

    # Save or show the plot
    if save_file:
        image_name = f"{image_dir}{snap.replace('.gz', '')}_sigma_{xlim}kpc_umask_b{blur}.png"
        plt.savefig(image_name)
        print("Image saved to", image_name)
    else:
        print("Image saving is turned off.")

    if show_plot:
        plt.show()
    else:
        print("On-screen plotting is turned off.")

    del vd_stat2d

    return fig


# Plots edge-on sigma for age groups
def plot_sigma_by_age(cmap,sim,xlim,ylim,bins,snap,image_dir,save_file=True,show_plot=True,verbose_log=False):

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    stat2d_lst = []

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
    
        #Extract phase space data from the model for the stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        vdg_stat2d,vdg_xedges,vdg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(vdg_stat2d.T)

    for i in range(x_panels):
                image = axes[i].imshow(stat2d_lst[i], 
                            origin = 'lower',
                            extent = [-xlim, xlim, -ylim, ylim ],
                            cmap = cmap)
                xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
                ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, stat2d_lst[i], linewidths = 2, linestyles = 'dashed', colors = 'w')
                axes[i].title.set_text("Age group " + str(i+1))
                # For nuclear bars teh circle R is 0.5 and 0.75 
                circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " LOS velocity dispersion.")
    plt.setp(axes[:], xlabel = 'y [kpc]')
    plt.setp(axes[0], ylabel = 'z [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_sigma_by_age_3grp_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots edge-on sigma for age groups using grouping switch
def plot_sigma_by_age_grp_sw(cmap,sim,snap,image_dir,save_file=True):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")
    
    # Check if min age is not equal to max age    
    check_ages(sim)
            
    # Number density statistics per age group
    age_grp = 0

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        if verbose_log:
            print_ages(grp,"group")
    
        #Extract phase space data from the model for the stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        vdg_stat2d,vdg_xedges,vdg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(vdg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                image = axes[i].imshow(stat2d_lst[i], 
                            origin = 'lower',
                            extent = [-xlim, xlim, -ylim, ylim ],
                            cmap = cmap)
                xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
                ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, stat2d_lst[i], linewidths = 2, linestyles = 'dashed', colors = 'w')
                axes[i].title.set_text("Age group " + str(i+1))
                # For nuclear bars teh circle R is 0.5 and 0.75 
                circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    suptit_name = snap.replace(".gz","") + " LOS velocity dispersio " + grp_sw 
    fig.suptitle(suptit_name)
    plt.setp(axes[:], xlabel = 'y [kpc]')
    plt.setp(axes[0], ylabel = 'z [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_sigma_by_age_3grp_' + str(xlim) + 'kpc_' + grp_sw + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots edge-on sigma for age groups using the unsharp mask
def plot_sigma_by_age_umask(cmap,sim,xlim,ylim,bins,blur,snap,image_dir,save_file=True,show_plot=True,verbose_log=False):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
    
        #Extract phase space data for the model for stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        vdg_stat2d,vdg_xedges,vdg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(vdg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                
                # Apply Unsharp Mask
                blurred = gf(stat2d_lst[i], sigma=blur)  # Apply Gaussian blur
                sharpened = stat2d_lst[i] + (stat2d_lst[i] - blurred)  # Unsharp masking

                # Plot the sharpened image
                image = axes[i].imshow(sharpened,
                      origin='lower',
                      extent=[-xlim, xlim, -ylim, ylim],
                      cmap=cmap)

                xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
                ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, stat2d_lst[i], linewidths = 2, linestyles = 'dashed', colors = 'w')
                axes[i].title.set_text("Age group " + str(i+1))
                # For nuclear bars teh circle R is 0.5 and 0.75 
                circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " LOS velocity dispersion (sharp " + str(blur) + ").")
    plt.setp(axes[:], xlabel = 'y [kpc]')
    plt.setp(axes[0], ylabel = 'z [kpc]')

    if save_file:
        image_name = f"{image_dir}{snap.replace('.gz', '')}_sigma_by_age_3grp_{xlim}kpc_umask_b{blur}.png"
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots edge-on sigma for age groups using grouping switch and unsharp mask
def plot_sigma_by_age_grp_sw_umask(cmap,sim,blur,snap,image_dir,save_file=True):

    stat2d_lst = []

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")
    
    # Check if min age is not equal to max age    
    check_ages(sim)
            
    # Number density statistics per age group
    age_grp = 0

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        if verbose_log:
            print_ages(grp,"group")
    
        #Extract phase space data for the model for stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']
    
        # Number density statistics face-on for stellar population by age group
        vdg_stat2d,vdg_xedges,vdg_yedges,df_binnum2d = st.binned_statistic_2d(x_, y_, vz_,
                                    statistic = 'std',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)
        stat2d_lst.append(vdg_stat2d.T)

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
                
                # Apply Unsharp Mask
                blurred = gf(stat2d_lst[i], sigma=blur)  # Apply Gaussian blur
                sharpened = stat2d_lst[i] + (stat2d_lst[i] - blurred)  # Unsharp masking

                # Plot the sharpened image
                image = axes[i].imshow(sharpened,
                      origin='lower',
                      extent=[-xlim, xlim, -ylim, ylim],
                      cmap=cmap)

                xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
                ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
                axes[i].contour(xcent, ycent, stat2d_lst[i], linewidths = 2, linestyles = 'dashed', colors = 'w')
                axes[i].title.set_text("Age group " + str(i+1))
                # For nuclear bars teh circle R is 0.5 and 0.75 
                circle1 = plt.Circle((0, 0), 3, color='green', fill=False)
                circle2 = plt.Circle((0, 0), 4, color='green', fill=False)
                axes[i].add_patch(circle1)
                axes[i].add_patch(circle2)

    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(image, cax=cax, orientation='vertical')
    #cbar.set_label(cbar_label_lst[i])
    if i > 0:
        axes[i].set_yticklabels([])

    fig.tight_layout()
    suptit_name = snap.replace(".gz","") + " LOS velocity dispersion " + grp_sw + " (sharp " + str(blur) + ")."
    fig.suptitle(suptit_name)
    plt.setp(axes[:], xlabel = 'y [kpc]')
    plt.setp(axes[0], ylabel = 'z [kpc]')

    if save_file:
        image_name = f"{image_dir}{snap.replace('.gz', '')}_sigma_by_age_3grp_{xlim}kpc_umask_b{blur}_{grp_sw}.png"
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")

    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del stat2d_lst

    return fig


# Plots edge-on sigma for age groups using grouping switch and unsharp mask (NaN-safe version)
# With fixed issued for color bar.
def plot_sigma_by_age_grp_sw_umask2(cmap, sim, blur, snap, image_dir, save_file=True):
    stat2d_lst = []

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim, "snapshot")

    # Check if min age is not equal to max age    
    check_ages(sim)

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        #Extract phase space data for the model for stars in the group
        vz_, x_, y_ = grp.star['vz'], grp.star['x'], grp.star['y']

        # Number density statistics face-on for stellar population by age group
        vdg_stat2d, vdg_xedges, vdg_yedges, _ = st.binned_statistic_2d(
            x_, y_, vz_,
            statistic='std',
            range=[[-xlim, xlim], [-ylim, ylim]],
            bins=bins
        )
        stat2d_lst.append(vdg_stat2d.T)

    # Setup for plotting
    x_panels, y_panels = 3, 1
    figsize_x, figsize_y = 3 * x_panels, 3.5 * y_panels
    fig, axes = plt.subplots(y_panels, x_panels, figsize=(figsize_x, figsize_y))

    # Compute global min and max across all age groups
    vmin = min(np.nanmin(d) for d in stat2d_lst)
    vmax = max(np.nanmax(d) for d in stat2d_lst)

    for i in range(x_panels):
        data = stat2d_lst[i]
        data_filled = np.nan_to_num(data, nan=np.nanmedian(data))  # Fill NaNs with median
        blurred = gf(data_filled, sigma=blur)
        sharpened = data_filled + (data_filled - blurred)  # Unsharp masking

        image = axes[i].imshow(
            sharpened,
            origin='lower',
            extent=[-xlim, xlim, -ylim, ylim],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        xcent = (vdg_xedges[1:] + vdg_xedges[:-1]) / 2
        ycent = (vdg_yedges[1:] + vdg_yedges[:-1]) / 2
        axes[i].contour(xcent, ycent, data, linewidths=2, linestyles='dashed', colors='w')
        axes[i].title.set_text("Age group " + str(i + 1))

        # Add reference circles
        axes[i].add_patch(plt.Circle((0, 0), 3, color='green', fill=False))
        axes[i].add_patch(plt.Circle((0, 0), 4, color='green', fill=False))

        if i > 0:
            axes[i].set_yticklabels([])

    # Add a shared colorbar after plotting all subplots
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(image, cax=cbar_ax)

    suptit_name = snap.replace(".gz", "") + " LOS velocity dispersion " + grp_sw + f" (sharp {blur})."
    fig.suptitle(suptit_name)
    plt.setp(axes[:], xlabel='y [kpc]')
    plt.setp(axes[0], ylabel='z [kpc]')

    if save_file:
        image_name = f"{image_dir}{snap.replace('.gz', '')}_sigma_by_age_3grp_{xlim}kpc_umask_b{blur}_{grp_sw}.png"
        plt.savefig(image_name)
        print("Image saved to", image_name)
    else:
        print("Image saving is turned off.")

    if show_plot:
        plt.show()
    else:
        print("On-screen plotting is turned off.")

    del stat2d_lst
    return fig


# Plots sigma shape estimation using amplitude and phase of Fourier moments m 4 or 6
def sigma_shape_Fm(sim,Fm,snap,image_dir,age_grp=0,save_file=True):
    
    bar_xlim = xlim + 0.5
    
    #Extract phase space data for the model
    x, y, z, vz = sim.s['x'], sim.s['y'], sim.s['z'], sim.s['vz']

    #Calculate the radius of each particle and its cylindrical angle phi
    Rs = np.hypot(x, z) # Hypotenuse of right angle triangle with x,z
    phis = np.arctan2(z, x) # in radians
    
    # Define bins for radius
    max_radius = np.max(Rs)
    num_radial_bins = int((max_radius) / bin_width) + 1
    radial_bins = np.arange(0, num_radial_bins * bin_width, bin_width)

    # Calculate arc bins from a pre-defined bin_arc in degrees
    arc_bins = int(360/bin_arc) + 1
    
    # Define bins for theta (angular bins) in radians
    phi_bins = np.linspace(0, 2 * np.pi, arc_bins)

    if verbose_log:
        # Debug output
        print("Radial bin width: ", bin_width, " kpc ", "Max R:", int(Rs.max()), " kpc -> number of radial bins:", num_radial_bins)
        print("Size of radial bins array", len(radial_bins))
        #print("Radial bins array: ",radial_bins)
        print("Arc bin width: ", bin_arc, " deg -> number of arc bins:", arc_bins)
        print("Size of phi bins array", len(phi_bins))
        #print("Phi bins array: ", phi_bins)
        print()
    
    # Assume x, y, vz are your numpy arrays for particle positions and velocities
    # radial_bins and phi_bins are numpy arrays defining the bin edges (left boundaries)

    # Step 1: Convert Cartesian coordinates to polar coordinates
    # See above - Rs and phis arrays
    #R_particles = np.sqrt(x**2 + y**2)               # Calculate R for each particle
    #Phi_particles = np.arctan2(y, x)                 # Calculate Phi for each particle

    # Step 2: Define the polar bins (radial_bins and phi_bins are provided)
    # radial_bins = np.array([...])
    # phi_bins = np.array([...])

    # Step 3: Identify the radial and angular bins for each particle

    # Find the radial bin indices
    radial_bin_indices = np.digitize(Rs, bins=radial_bins) - 1
    # `np.digitize` returns the index of the bin to which each value belongs

    # Find the angular bin indices
    phi_bin_indices = np.digitize(phis, bins=phi_bins) - 1

    # Handle periodic boundary condition for angles (wrapping around 2π)
    Phi_particles_wrapped = (phis + 2 * np.pi) % (2 * np.pi)
    phi_bin_indices_wrapped = np.digitize(Phi_particles_wrapped, bins=phi_bins) - 1

    # Step 4: Initialize a 2D array to store arrays for each bin
    num_radial_bins = len(radial_bins) - 1
    num_phi_bins = len(phi_bins) - 1
    num_values_bins = 4 # x, y, vz, sigma

    # Initialize the 2D array with empty sub-arrays
    particles_in_bins = np.empty((num_radial_bins, num_phi_bins, num_values_bins), dtype=object)

    for i in range(num_radial_bins):
        for j in range(num_phi_bins):
            for k in range(num_values_bins):
                if k == 3:
                    particles_in_bins[i, j, k] = nan
                else:
                    particles_in_bins[i, j, k] = []

    for i in range(len(radial_bins) - 1):  # Iterate through radial bins
        for j in range(len(phi_bins) - 1):  # Iterate through angular bins
            # Select particles in the current bin
            in_radial_bin = radial_bin_indices == i
            in_angular_bin = (phi_bin_indices == j) | (phi_bin_indices_wrapped == j)

            # Combine conditions
            inside_bin = in_radial_bin & in_angular_bin

            # Get Cartesian coordinates and vz for particles in this bin
            particles_in_bins[i, j, 0] = x[inside_bin]
            particles_in_bins[i, j, 1] = y[inside_bin]
            particles_in_bins[i, j, 2] = vz[inside_bin]
            particles_in_bins[i, j, 3] = np.std(vz[inside_bin]) # sigma for the bin

            #if verbose_log:
            #   Debug output
            #   print(f"Radial bin {i}, Angular Bin {j}, Number of particles {len(particles_in_bins[i, j, 2])}, Std Dev = {particles_in_bins[i,j,3]:.2f}")

    # particles_in_bins now contains lists of particles (with Cartesian coordinates and vz)
    # for each radial and angular bin combination

    # Remove the last eement of each array to match the other arrays for broadcast operations
    if verbose_log:
        # Debug output
        print("Removing last elements of phi bins and radial bins arrays")
        print("to match with other arrays for broadcast operations.")
        print()
    
    max_size = len(phi_bins)
    last_index = max_size - 1
    phi_bins = np.delete(phi_bins,last_index)

    max_size = len(radial_bins)
    last_index = max_size - 1
    radial_bins = np.delete(radial_bins,last_index)
    
    if verbose_log:
        # Debug output
        print("New size of radial bins array", len(radial_bins))
        print("New size of phi bins array", len(phi_bins))

    # for each arc bin with arc "bin_arc" and width "bin_width"
    sFp = np.sin(Fm * (phi_bins + bin_arc/2))
    cFp = np.cos(Fm * (phi_bins + bin_arc/2))

    if verbose_log:
        # Debug output
        print("Sigma Fourier moment Fm:", Fm)
        print("Size of sine(Fm * phi) array;", len(sFp))
        print("Size of cosine(Fm * phi) arra:", len(cFp))
        print()

    aF_plot = []
    phiF_plot = []

    # Calculate AMPLITUDE and PHASE per radius
    # by summing across all phi bins for each radial bin
    # To be corrected - "range(len(radial_bins) - 1" or "range(len(radial_bins)"
    for i in range(len(radial_bins)):
        aF = hypot(np.sum(sFp*particles_in_bins[i,:,3]),np.sum(cFp*particles_in_bins[i,:,3]))/np.sum(particles_in_bins[i,:,3])
        phiF = (1/Fm) * degrees(arctan2(np.sum(sFp*particles_in_bins[i,:,3]),np.sum(cFp*particles_in_bins[i,:,3])))
        
        aF_plot.append(aF)
        phiF_plot.append(phiF)

    if verbose_log:
        # Debug output
        print("Size of AMPLITUDE array", len(aF_plot))
        print("Min value of AMPLITUDE array", round(np.nanmin(aF_plot),2))
        print("Max value of AMPLITUDE array", round(np.nanmax(aF_plot),2))
        #print("All values of AMPLITUDE array", aF_plot)
        print("Size of PHASE array", len(phiF_plot))
        print("Min value of PHASE array", round(np.nanmin(phiF_plot),2))
        print("Max value of PHASE array", round(np.nanmax(phiF_plot),2))
        #print("All values of PHASE array", phiF_plot)
        print()

    # Plot the amplitude diagram
    xlab = r'$R \rm \enspace [kpc]$'
    ylab = r'$A(R)$'
    y2lab = r'$\phi(R) \enspace \rm [deg.]$'

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    ax = axes
    ax2 = ax.twinx()
    fs = 8

    ax.plot(radial_bins, aF_plot, c='r', label='r$A(R)$')
    ax2.plot(radial_bins, phiF_plot, c='b', label='r$\phi(R)$')
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax2.tick_params(axis='both', which='both', labelsize=fs)
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs, c='r')
    ax2.set_ylabel(y2lab, fontsize=fs, c='b')
    # We do not plot the analysis conditions for bars from Stuart code for sigma right now.
    #ax.axvline(bar_ends_R_aF, c='r', ls='--')
    #ax.axvline(bar_ends_R_phiF, c='b', ls='--')
    ax.set_xlim(0., bar_xlim)
    
    title = snap.replace(".gz","") + " Amp-Phase Fm " + str(Fm)
    if age_grp != 0:
        title = title + " age group " + str(age_grp) 
    ax.title.set_text(title)

    age_group = ""
    if age_grp != 0:
        age_group = f"age_grp_{str(age_grp)}_{grp_sw}_"

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_sigma_shape_' + age_group + "_Fm_" + str(Fm) + "_" + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    return None


# Wrap function for age groups, calls sigma_shape_Fm
def sigma_shape_by_age_Fm(sim,Fm,snap,image_dir,save_file=True):

    # Divide snapshot into 3 age groups
    min_age = round(min(sim.star['age']),2)
    max_age = round(max(sim.star['age']),2)
    if verbose_log:
        print('* Min stellar age - ' + str(min_age) + ' Gyr.')
        print('* Max stellar age - ' + str(max_age) + ' Gyr.')
        print('** Stars in snapshot - ', len(sim.star))
    if min_age == max_age:
        print("** Cannot split into age groups for this snapshot.")
        print("** The min age and the max age of the stars are the same.")
        exit
            
    # Number density statistics per age group
    div = 1/3
    divlr_lst = [[0,div],[div,2*div],[2*div,max_age]]
    age_grp = 0

    for divlr in divlr_lst:
        age_grp += 1
        div_l = divlr[0]
        div_r = divlr[1]
        # Mask stars between age dividers div_l and div_r
        mask = ma.masked_inside(sim.star['age'], max_age*div_l, max_age*div_r).mask
        # print(len(mask))
        grp = sim.star[mask]
        if verbose_log:
        # Debug output
            print('*** Stars in age group', age_grp, ' - ', len(grp.star['age']))
    
        sigma_shape_Fm(grp,Fm,snap,image_dir,age_grp,save_file)

    return None


# Wrap function for age groups using grouping switch, calls sigma_shape_Fm
def sigma_shape_by_age_grp_sw_Fm(sim,Fm,snap,image_dir,save_file=True):

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")

    # Check if min age is not equal to max age    
    check_ages(sim)
            
    # Number density statistics per age group
    age_grp = 0

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        if verbose_log:
            print_ages(grp,"group")
    
        sigma_shape_Fm(grp,Fm,snap,image_dir,age_grp,save_file)

    return None


# Plots sigma shape estimation using gropuping switch, amplitude and phase of Fourier moments m 4 or 6
def sigma_shape_by_age_grp_sw_combined_Fm(sim,sigmaFm,snap,image_dir,age_grp=0,save_file=True):
    
    if verbose_log:
        print('!!! Function starts executiung here !!!') 

    # Divide snapshot into 3 age groups
    if verbose_log:
        print_ages(sim,"snapshot")

    max_age = round(max(sim.star['age']),2)

    # Check if min age is not equal to max age
    check_ages(sim)
            
    aF_plot_age_grp = []
    phiF_plot_age_grp = []
    radial_bins_age_grp = []
    aF_max_age_grp = []
    aF_max_R_age_grp = []

    # Number density statistics per age group
    age_grp = 0

    # Call the grouping function 
    groups = grouping(sim)

    for grp in groups:
        age_grp += 1
        if verbose_log:
            print_ages(grp,"group")

        #Extract phase space data for the model
        x, y, z, vz = grp.s['x'], grp.s['y'], grp.s['z'], grp.s['vz']

        #Calculate the radius of each particle and its cylindrical angle phi
        Rs = np.hypot(x, z) # Hypotenuse of right angle triangle with x,z
        phis = np.arctan2(z, x) # in radians

        # Define bins for radius
        max_radius = np.max(Rs)
        num_radial_bins = int((max_radius) / bin_width) + 1
        radial_bins = np.arange(0, num_radial_bins * bin_width, bin_width)
        if verbose_log:
            print(radial_bins)

        # Calculate arc bins from a pre-defined bin_arc in degrees
        arc_bins = int(360/bin_arc) + 1

        # Define bins for theta (angular bins) in radians
        phi_bins = np.linspace(0, 2 * np.pi, arc_bins)

        if verbose_log:
            # Debug output
            print("Radial bin width: ", bin_width, " kpc ", "Max R:", int(Rs.max()), " kpc -> number of radial bins:", num_radial_bins)
            print("Size of radial bins array", len(radial_bins))
            #print("Radial bins array: ",radial_bins)
            print("Arc bin width: ", bin_arc, " deg -> number of arc bins:", arc_bins)
            print("Size of phi bins array", len(phi_bins))
            #print("Phi bins array: ", phi_bins)
            print()

        if verbose_log:
            # Debug output
            print("!!! Start calculating Cartesian coordinates for radial bins !!!")

        # Assume x, y, vz are your numpy arrays for particle positions and velocities
        # radial_bins and phi_bins are numpy arrays defining the bin edges (left boundaries)

        # Step 1: Convert Cartesian coordinates to polar coordinates
        # See above - Rs and phis arrays 
        #R_particles = np.sqrt(x**2 + y**2)               # Calculate R for each particle
        #Phi_particles = np.arctan2(y, x)                 # Calculate Phi for each particle

        # Step 2: Define the polar bins (radial_bins and phi_bins are provided)
        # radial_bins = np.array([...])
        # phi_bins = np.array([...])

        # Step 3: Identify the radial and angular bins for each particle

        # Find the radial bin indices
        radial_bin_indices = np.digitize(Rs, bins=radial_bins) - 1
        # `np.digitize` returns the index of the bin to which each value belongs

        # Find the angular bin indices
        phi_bin_indices = np.digitize(phis, bins=phi_bins) - 1

        # Handle periodic boundary condition for angles (wrapping around 2π)
        Phi_particles_wrapped = (phis + 2 * np.pi) % (2 * np.pi)
        phi_bin_indices_wrapped = np.digitize(Phi_particles_wrapped, bins=phi_bins) - 1

        # Step 4: Initialize a 2D array to store arrays for each bin
        num_radial_bins = len(radial_bins) - 1
        num_phi_bins = len(phi_bins) - 1
        num_values_bins = 4 # x, y, vz, sigma

        # Initialize the 2D array with empty sub-arrays
        particles_in_bins = np.empty((num_radial_bins, num_phi_bins, num_values_bins), dtype=object)

        for i in range(num_radial_bins):
            for j in range(num_phi_bins):
                for k in range(num_values_bins):
                    if k == 3:
                        particles_in_bins[i, j, k] = nan
                    else:
                        particles_in_bins[i, j, k] = []

        for i in range(len(radial_bins) - 1):  # Iterate through radial bins
            for j in range(len(phi_bins) - 1):  # Iterate through angular bins
                # Select particles in the current bin
                in_radial_bin = radial_bin_indices == i
                in_angular_bin = (phi_bin_indices == j) | (phi_bin_indices_wrapped == j)

                # Combine conditions
                inside_bin = in_radial_bin & in_angular_bin

                # Get Cartesian coordinates and vz for particles in this bin
                particles_in_bins[i, j, 0] = x[inside_bin]
                particles_in_bins[i, j, 1] = y[inside_bin]
                particles_in_bins[i, j, 2] = vz[inside_bin]
                particles_in_bins[i, j, 3] = np.std(vz[inside_bin]) # sigma for the bin

                #if verbose_log:
                #   Debug output
                #   print(f"Radial bin {i}, Angular Bin {j}, Number of particles {len(particles_in_bins[i, j, 2])}, Std Dev = {particles_in_bins[i,j,3]:.2f}")

        # particles_in_bins now contains lists of particles (with Cartesian coordinates and vz)
        # for each radial and angular bin combination

        # Remove the last eement of each array to match the other arrays for broadcast operations
        if verbose_log:
            # Debug output
            print("Removing last elements of phi bins and radial bins arrays")
            print("to match with other arrays for broadcast operations.")
            print()

        max_size = len(phi_bins)
        last_index = max_size - 1
        phi_bins = np.delete(phi_bins,last_index)

        max_size = len(radial_bins)
        last_index = max_size - 1
        radial_bins = np.delete(radial_bins,last_index)

        if verbose_log:
            # Debug output
            print("New size of radial bins array", len(radial_bins))
            print("New size of phi bins array", len(phi_bins))

        aF_plot_comb = []   # Amplitude for Fourier moments m=4 and m=6
        phiF_plot_comb = [] # Phase for Fourier moments m=4 and m=6
        aF_max_comb = []    # Max amplitude for Fourier moments m=4 and m=6
        aF_max_R_comb = []  # Max amplitude radii for Fourier moments m=4 and m=6

        for Fm in sigmaFm: # !!! TBD - implement as a stanalone function !!!

            if verbose_log:
                print('!!! Start calculating and using Fourier moments !!!')

            # for each arc bin with arc "bin_arc" and width "bin_width"
            sFp = np.sin(Fm * (phi_bins + bin_arc/2))
            cFp = np.cos(Fm * (phi_bins + bin_arc/2))

            if verbose_log:
                # Debug output
                print("Sigma Fourier moment Fm:", Fm)
                print("Size of sine(Fm * phi) array;", len(sFp))
                print("Size of cosine(Fm * phi) arra:", len(cFp))
                print()

            aF_plot = []
            phiF_plot = []

            # Calculate AMPLITUDE and PHASE per radius
            # by summing across all phi bins for each radial bin
            # To be corrected - "range(len(radial_bins) - 1" or "range(len(radial_bins)"
            for i in range(len(radial_bins)):
                aF = hypot(np.sum(sFp*particles_in_bins[i,:,3]),np.sum(cFp*particles_in_bins[i,:,3]))/np.sum(particles_in_bins[i,:,3])
                phiF = (1/Fm) * degrees(arctan2(np.sum(sFp*particles_in_bins[i,:,3]),np.sum(cFp*particles_in_bins[i,:,3])))

                aF_plot.append(aF)
                phiF_plot.append(phiF)

            if verbose_log:
                # Debug output
                print("Size of AMPLITUDE array", len(aF_plot))
                #print("Min value of AMPLITUDE array", round(np.nanmin(aF_plot),2))
                print("Max value of AMPLITUDE array", round(np.nanmax(aF_plot),2))
                #print("All values of AMPLITUDE array", aF_plot)
                #print("Size of PHASE array", len(phiF_plot))
                #print("Min value of PHASE array", round(np.nanmin(phiF_plot),2))
                #print("Max value of PHASE array", round(np.nanmax(phiF_plot),2))
                #print("All values of PHASE array", phiF_plot)
                print()

            # aF_peaks, _ = find_peaks(aF_plot)
            #aF_max = aF_plot[aF_peaks[0]]
            #aF_max_R = radial_bins[aF_peaks[0]]
            idx_1kpc = np.where(radial_bins == 1.)[0][0]
            aF_plot_1kpc = aF_plot[:idx_1kpc]
            aF_max = np.nanmax(aF_plot_1kpc)
            aF_max_index = aF_plot.index(aF_max)
            aF_max_R = radial_bins[aF_max_index]

            print("Age group", age_grp, "- Fm", Fm, "- AMPLITUDE peak in [", -xlim, ":", xlim, "] kpc area is", round(aF_max,2), "at radius", round(aF_max_R,1), "kpc.")

            aF_plot_comb.append(aF_plot)
            phiF_plot_comb.append(phiF_plot)

            aF_max_comb.append(aF_max)
            aF_max_R_comb.append(aF_max_R)

        if verbose_log:
            # Debug output
            print("Size of AMPLITUDE combined array", len(aF_plot_comb))
            print()
            print()

        aF_plot_age_grp.append(aF_plot_comb)
        phiF_plot_age_grp.append(phiF_plot_comb)
        radial_bins_age_grp.append(radial_bins)
        aF_max_age_grp.append(aF_max_comb)
        aF_max_R_age_grp.append(aF_max_R_comb)

    if verbose_log:
        # Debug output
        print("Size of AMPLITUDE array for all age groups", len(aF_plot_age_grp))

    # Plot the combined amplitude diagram for Fm 4 and 6
    xlab = r'$R \rm \enspace [kpc]$'
    ylab = r'$A(R)_Fm=4 $'
    y2lab = r'$A(R)_Fm=6 $'

    x_panels = 3
    y_panels = 1

    figsize_x = 3*x_panels      # inches
    figsize_y = 3.5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    for i in range(x_panels):
        
        ax = axes[i]
        ax2 = ax.twinx()
        fs = 8

        # Unpack values for plotting from arrays
        radial_bins = radial_bins_age_grp[i]
        aF_plot_comb = aF_plot_age_grp[i]
        aF_plot_4 = aF_plot_comb[0]
        aF_plot_6 = aF_plot_comb[1]

        aF_max_R_comb = aF_max_R_age_grp[i]
        X_ends_R_aF4 = aF_max_R_comb[0]
        X_ends_R_aF6 = aF_max_R_comb[1]

        ax.plot(radial_bins, aF_plot_4, c='r', label='r$A(R)$ Fm=4')
        ax2.plot(radial_bins, aF_plot_6, c='b', label='r$A(R)$ Fm=6')
        ax.tick_params(axis='both', which='both', labelsize=fs)
        ax2.tick_params(axis='both', which='both', labelsize=fs)
        ax.set_xlabel(xlab, fontsize=fs)
        ax.set_ylabel(ylab, fontsize=fs, c='r')
        ax2.set_ylabel(y2lab, fontsize=fs, c='b')
        # We do not plot the analysis conditions for bars from Stuart code for sigma right now.
        ax.axvline(X_ends_R_aF4, c='r', ls='-')
        ax.axvline(X_ends_R_aF6, c='b', ls='--')
        ax.set_xlim(0., xlim)

        title = "Age group " + str(i + 1) 
        ax.title.set_text(title)
        ax.title.set_size(fs)

        #divider = make_axes_locatable(axes[i])
        #cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.tight_layout()
    suptit_name = snap.replace(".gz","") + " sigma amplitude per age group " + grp_sw + " Fm 4 and 6."
    fig.suptitle(suptit_name, fontsize=fs)  
    #plt.setp(axes[:], xlabel = 'x [kpc]')
    #plt.setp(axes[0], ylabel = 'y [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_sigma_amp_comb_by_age_3grp_' + str(xlim) + 'kpc_' + grp_sw + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del aF_plot_age_grp
    del phiF_plot_age_grp

    return fig, aF_max_age_grp, max_age


# Plot sigma amplitude and ellipticity timelies across all snapshots of the model
def plot_sigma_amp_bar_ellip_timeline(model,aF_peaks,e_list,snap_ages,plot_bar_ellipticity,plot_sigma_amp,image_dir,save_file=True):
    # Plot the sigma amplitude and bar ellipticity diagram
    xlab = 'Age [Gyr]'
    if plot_sigma_amp: ylab = 'Sigma amplitude peaks'
    if plot_bar_ellipticity: y2lab = 'Bar ellipticities'

    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    ax = axes
    ax2 = ax.twinx()
    fs = 8
    fs_l = 6

    if plot_sigma_amp:
        aF_peaks_grp1_m4 = []
        aF_peaks_grp1_m6 = []
        aF_peaks_grp2_m4 = []
        aF_peaks_grp2_m6 = []
        aF_peaks_grp3_m4 = []
        aF_peaks_grp3_m6 = []
    
        # Unpack arrays for plotting
        for elem in aF_peaks:
            aF_peaks_grp1_m4.append(elem[0][0])
            aF_peaks_grp1_m6.append(elem[0][1])
            aF_peaks_grp2_m4.append(elem[1][0])
            aF_peaks_grp2_m6.append(elem[1][1])
            aF_peaks_grp3_m4.append(elem[2][0])
            aF_peaks_grp3_m6.append(elem[2][1])

        ax.plot(snap_ages, aF_peaks_grp1_m4, c='r', linestyle='-', label='Group 1 Fm=4')
        ax.plot(snap_ages, aF_peaks_grp1_m6, c='r', linestyle='-.', label='Group 1 Fm=6')
        ax.plot(snap_ages, aF_peaks_grp2_m4, c='g', linestyle='-', label='Group 2 Fm=4')
        ax.plot(snap_ages, aF_peaks_grp2_m6, c='g', linestyle='-.', label='Group 2 Fm=6')
        ax.plot(snap_ages, aF_peaks_grp3_m4, c='b', linestyle='-', label='Group 3 Fm=4')
        ax.plot(snap_ages, aF_peaks_grp3_m6, c='b', linestyle='-.', label='Group 3 Fm=6')
        ax.tick_params(axis='both', which='both', labelsize=fs)
        ax.set_ylabel(ylab, fontsize=fs, c='k')
        ax.legend(loc="upper left")

    if plot_bar_ellipticity:
        e_list_grp1 = []
        e_list_grp2 = []
        e_list_grp3 = []

        for elem in e_list:
            e_list_grp1.append(elem[0])
            e_list_grp2.append(elem[1])
            e_list_grp3.append(elem[2])

        ax2.plot(snap_ages, e_list_grp1, c='r', label='Group 1 bar ellipticity', linestyle=':')
        ax2.plot(snap_ages, e_list_grp2, c='g', label='Group 2 bar ellipticity', linestyle=':')
        ax2.plot(snap_ages, e_list_grp3, c='b', label='Group 3 bar ellipticity', linestyle=':')
        ax2.tick_params(axis='both', which='both', labelsize=fs)
        ax2.set_ylabel(y2lab, fontsize=fs, c='k')
        ax2.legend(loc="upper right")
        #ax2.axis('equal')

    #plt.legend()
    #ax.axis('equal')
    #leg = ax.legend(loc="upper left")

    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_xlim(np.nanmin(snap_ages), np.nanmax(snap_ages))
    if plot_bar_ellipticity and plot_sigma_amp:
        title = model.replace("run","") + " sigma amplitude peaks and bar ellipticities evolution"
    elif plot_bar_ellipticity:
        title = model.replace("run","") + " bar ellipticities evolution"
    elif plot_sigma_amp:
        title = model.replace("run","") + " sigma amplitude peaks evolution"
    else:
        title = model.replace("run","") + " WTF evolution"
    ax.title.set_text(title)

    if save_file:
        if plot_bar_ellipticity and plot_sigma_amp:
            image_name = image_dir + model.replace("run","") + '_sigma_amp_bar_ellip.png'
        elif plot_bar_ellipticity:
            image_name = image_dir + model.replace("run","") + '_bar_ellip.png'
        elif plot_sigma_amp:
            image_name = image_dir + model.replace("run","") + '_sigma_amp.png'
        else:
            image_name = image_dir + model.replace("run","") + '_WTF.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    return None


# Plots sigma shape estimation using amplitude and phase of Fourier moments m 4 or 6
def sigma_shape_combined_Fm(sim,sigmaFm,snap,image_dir,age_grp=0,save_file=True):
    
    if verbose_log:
        print('!!! Function starts executiung here !!!') 

    x_panels = 1
    y_panels = 1

    figsize_x = 12*x_panels   # inches
    figsize_y = 5*y_panels    # inches

    # Make the figure and sub plots
    fig,axes = plt.subplots(y_panels,x_panels,figsize=(figsize_x,figsize_y))
    
    max_age = round(max(sim.star['age']),2)

    aF_plot = []
    phiF_plot = []
    radial_bins = []
    aF_max = []
    aF_max_R = []

    #Extract phase space data for the model
    x, y, z, vz = sim.s['x'], sim.s['y'], sim.s['z'], sim.s['vz']

    #Calculate the radius of each particle and its cylindrical angle phi
    Rs = np.hypot(x, z) # Hypotenuse of right angle triangle with x,z
    phis = np.arctan2(z, x) # in radians

    # Define bins for radius
    max_radius = np.max(Rs)
    num_radial_bins = int((max_radius) / bin_width) + 1
    radial_bins = np.arange(0, num_radial_bins * bin_width, bin_width)
    if verbose_log:
        print(radial_bins)

    # Calculate arc bins from a pre-defined bin_arc in degrees
    arc_bins = int(360/bin_arc) + 1

    # Define bins for theta (angular bins) in radians
    phi_bins = np.linspace(0, 2 * np.pi, arc_bins)

    if verbose_log:
            # Debug output
            print("Radial bin width: ", bin_width, " kpc ", "Max R:", int(Rs.max()), " kpc -> number of radial bins:", num_radial_bins)
            print("Size of radial bins array", len(radial_bins))
            #print("Radial bins array: ",radial_bins)
            print("Arc bin width: ", bin_arc, " deg -> number of arc bins:", arc_bins)
            print("Size of phi bins array", len(phi_bins))
            #print("Phi bins array: ", phi_bins)
            print()

    if verbose_log:
            # Debug output
            print("!!! Start calculating Cartesian coordinates for radial bins !!!")

        # Find the radial bin indices
    radial_bin_indices = np.digitize(Rs, bins=radial_bins) - 1
        # `np.digitize` returns the index of the bin to which each value belongs

        # Find the angular bin indices
    phi_bin_indices = np.digitize(phis, bins=phi_bins) - 1

        # Handle periodic boundary condition for angles (wrapping around 2π)
    Phi_particles_wrapped = (phis + 2 * np.pi) % (2 * np.pi)
    phi_bin_indices_wrapped = np.digitize(Phi_particles_wrapped, bins=phi_bins) - 1

        # Step 4: Initialize a 2D array to store arrays for each bin
    num_radial_bins = len(radial_bins) - 1
    num_phi_bins = len(phi_bins) - 1
    num_values_bins = 4 # x, y, vz, sigma

        # Initialize the 2D array with empty sub-arrays
    particles_in_bins = np.empty((num_radial_bins, num_phi_bins, num_values_bins), dtype=object)

    for i in range(num_radial_bins):
        for j in range(num_phi_bins):
            for k in range(num_values_bins):
                if k == 3:
                    particles_in_bins[i, j, k] = nan
                else:
                    particles_in_bins[i, j, k] = []

    for i in range(len(radial_bins) - 1):  # Iterate through radial bins
        for j in range(len(phi_bins) - 1):  # Iterate through angular bins
            # Select particles in the current bin
            in_radial_bin = radial_bin_indices == i
            in_angular_bin = (phi_bin_indices == j) | (phi_bin_indices_wrapped == j)

                # Combine conditions
            inside_bin = in_radial_bin & in_angular_bin

                # Get Cartesian coordinates and vz for particles in this bin
            particles_in_bins[i, j, 0] = x[inside_bin]
            particles_in_bins[i, j, 1] = y[inside_bin]
            particles_in_bins[i, j, 2] = vz[inside_bin]
            particles_in_bins[i, j, 3] = np.std(vz[inside_bin]) # sigma for the bin

                #if verbose_log:
                #   Debug output
                #   print(f"Radial bin {i}, Angular Bin {j}, Number of particles {len(particles_in_bins[i, j, 2])}, Std Dev = {particles_in_bins[i,j,3]:.2f}")

        # particles_in_bins now contains lists of particles (with Cartesian coordinates and vz)
        # for each radial and angular bin combination

        # Remove the last eement to match the other arrays for broadcast operations
    if verbose_log:
            # Debug output
            print("Removing last elements of phi bins and radial bins arrays")
            print("to match with other arrays for broadcast operations.")
            print()

    max_size = len(phi_bins)
    last_index = max_size - 1
    phi_bins = np.delete(phi_bins,last_index)

    max_size = len(radial_bins)
    last_index = max_size - 1
    radial_bins = np.delete(radial_bins,last_index)

    if verbose_log:
            # Debug output
            print("New size of radial bins array", len(radial_bins))
            print("New size of phi bins array", len(phi_bins))

    aF_plot_comb = []   # Amplitude for Fourier moments m=4 and m=6
    phiF_plot_comb = [] # Phase for Fourier moments m=4 and m=6
    aF_max_comb = []    # Max amplitude for Fourier moments m=4 and m=6
    aF_max_R_comb = []  # Max amplitude radii for Fourier moments m=4 and m=6

    for Fm in sigmaFm: # !!! TBD - implement as a stanalone function !!!

        if verbose_log:
                print('!!! Start calculating and using Fourier moments !!!')

            # for each arc bin with arc "bin_arc" and width "bin_width"
        sFp = np.sin(Fm * (phi_bins + bin_arc/2))
        cFp = np.cos(Fm * (phi_bins + bin_arc/2))

        if verbose_log:
                # Debug output
                print("Sigma Fourier moment Fm:", Fm)
                print("Size of sine(Fm * phi) array;", len(sFp))
                print("Size of cosine(Fm * phi) arra:", len(cFp))
                print()

        aF_plot = []
        phiF_plot = []

            # Calculate AMPLITUDE and PHASE per radius
            # by summing across all phi bins for each radial bin
            # To be corrected - "range(len(radial_bins) - 1" or "range(len(radial_bins)"
        for i in range(len(radial_bins)):
            aF = hypot(np.sum(sFp*particles_in_bins[i,:,3]),np.sum(cFp*particles_in_bins[i,:,3]))/np.sum(particles_in_bins[i,:,3])
            phiF = (1/Fm) * degrees(arctan2(np.sum(sFp*particles_in_bins[i,:,3]),np.sum(cFp*particles_in_bins[i,:,3])))

            aF_plot.append(aF)
            phiF_plot.append(phiF)

        if verbose_log:
                # Debug output
                print("Size of AMPLITUDE array", len(aF_plot))
                print("Max value of AMPLITUDE array", round(np.nanmax(aF_plot),2))
                print()

            # aF_peaks, _ = find_peaks(aF_plot)
            #aF_max = aF_plot[aF_peaks[0]]
            #aF_max_R = radial_bins[aF_peaks[0]]
        idx_1kpc = np.where(radial_bins == 1.)[0][0]
        aF_plot_1kpc = aF_plot[:idx_1kpc]
        aF_max = np.nanmax(aF_plot_1kpc)
        aF_max_index = aF_plot.index(aF_max)
        aF_max_R = radial_bins[aF_max_index]

        print("Fourier moment", Fm, "- AMPLITUDE peak in [", -xlim, ":", xlim, "] kpc area is", round(aF_max,2), "at radius", round(aF_max_R,1), "kpc.")

        aF_plot_comb.append(aF_plot)
        phiF_plot_comb.append(phiF_plot)

        aF_max_comb.append(aF_max)
        aF_max_R_comb.append(aF_max_R)

    if verbose_log:
            # Debug output
            print()
            print("Start of berbose log")
            print()
            print("Size of AMPLITUDE combined array", len(aF_plot_comb))
            print()
            print("End of berbose log")
            print()

    #aF_plot.append(aF_plot_comb)
    #phiF_plot.append(phiF_plot_comb)
    #radial_bins.append(radial_bins)
    #aF_max.append(aF_max_comb)
    #aF_max_R.append(aF_max_R_comb)

    if verbose_log:
        # Debug output
        print()
        print("Start of berbose log")
        print()
        print("Size of AMPLITUDE array for all age groups", len(aF_plot))
        print()
        print("End of berbose log")
        print()

    # Plot the combined amplitude diagram for Fm 4 and 6
    xlab = r'$R \rm \enspace [kpc]$'
    ylab = r'$A(R)_Fm=4 $'
    y2lab = r'$A(R)_Fm=6 $'
    
    ax = axes
    ax2 = axes.twinx()
    fs = 8

        # Unpack values for plotting from arrays
        #radial_bins = radial_bins[i]
        #aF_plot_comb = aF_plot[i]
    aF_plot_4 = aF_plot_comb[0]
    aF_plot_6 = aF_plot_comb[1]

        #aF_max_R_comb = aF_max_R[i]
    X_ends_R_aF4 = aF_max_R_comb[0]
    X_ends_R_aF6 = aF_max_R_comb[1]

    ax.plot(radial_bins, aF_plot_4, c='r', label='r$A(R)$ Fm=4')
    ax2.plot(radial_bins, aF_plot_6, c='b', label='r$A(R)$ Fm=6')
    ax.tick_params(axis='both', which='both', labelsize=fs)
    ax2.tick_params(axis='both', which='both', labelsize=fs)
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs, c='r')
    ax2.set_ylabel(y2lab, fontsize=fs, c='b')
    # We do not plot the analysis conditions for bars from Stuart code for sigma right now.
    ax.axvline(X_ends_R_aF4, c='r', ls='-')
    ax.axvline(X_ends_R_aF6, c='b', ls='--')
    ax.set_xlim(0., xlim)

    #divider = make_axes_locatable(axes[i])
    #cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.tight_layout()
    fig.suptitle(snap.replace(".gz","") + " sigma amplitude, Fourier moments 4 and 6.", fontsize=fs)
    #plt.setp(axes[:], xlabel = 'x [kpc]')
    #plt.setp(axes[0], ylabel = 'y [kpc]')

    if save_file:
        image_name = image_dir + snap.replace(".gz","") + '_sigma_amp_comb_' + str(xlim) + 'kpc' + '.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    del aF_plot
    del phiF_plot

    return fig, aF_max_comb, max_age


# Plot sigma amplitude and bar ellipticity timeline for all snapshots of the simulation
def plot_sigma_amp_bar_ellip_timeline(model,aF_peaks,e_list,snap_ages,plot_bar_ellipticity,plot_sigma_amp,image_dir,save_file=True):
    # Plot the sigma amplitude and bar ellipticity diagram
    xlab = 'Age [Gyr]'
    if plot_sigma_amp: ylab = 'Sigma amplitude peaks'
    if plot_bar_ellipticity: y2lab = 'Bar ellipticities'

    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    ax = axes
    ax2 = ax.twinx()
    fs = 8
    fs_l = 6

    if plot_sigma_amp:
        aF_peaks_grp1_m4 = []
        aF_peaks_grp1_m6 = []
        aF_peaks_grp2_m4 = []
        aF_peaks_grp2_m6 = []
        aF_peaks_grp3_m4 = []
        aF_peaks_grp3_m6 = []
    
        # Unpack arrays for plotting
        for elem in aF_peaks:
            aF_peaks_grp1_m4.append(elem[0][0])
            aF_peaks_grp1_m6.append(elem[0][1])
            aF_peaks_grp2_m4.append(elem[1][0])
            aF_peaks_grp2_m6.append(elem[1][1])
            aF_peaks_grp3_m4.append(elem[2][0])
            aF_peaks_grp3_m6.append(elem[2][1])

        ax.plot(snap_ages, aF_peaks_grp1_m4, c='r', linestyle='-', label='Group 1 Fm=4')
        ax.plot(snap_ages, aF_peaks_grp1_m6, c='r', linestyle='-.', label='Group 1 Fm=6')
        ax.plot(snap_ages, aF_peaks_grp2_m4, c='g', linestyle='-', label='Group 2 Fm=4')
        ax.plot(snap_ages, aF_peaks_grp2_m6, c='g', linestyle='-.', label='Group 2 Fm=6')
        ax.plot(snap_ages, aF_peaks_grp3_m4, c='b', linestyle='-', label='Group 3 Fm=4')
        ax.plot(snap_ages, aF_peaks_grp3_m6, c='b', linestyle='-.', label='Group 3 Fm=6')
        ax.tick_params(axis='both', which='both', labelsize=fs)
        ax.set_ylabel(ylab, fontsize=fs, c='k')
        ax.legend(loc="upper left")

    if plot_bar_ellipticity:
        e_list_grp1 = []
        e_list_grp2 = []
        e_list_grp3 = []

        for elem in e_list:
            e_list_grp1.append(elem[0])
            e_list_grp2.append(elem[1])
            e_list_grp3.append(elem[2])

        ax2.plot(snap_ages, e_list_grp1, c='r', label='Group 1 bar ellipticity', linestyle=':')
        ax2.plot(snap_ages, e_list_grp2, c='g', label='Group 2 bar ellipticity', linestyle=':')
        ax2.plot(snap_ages, e_list_grp3, c='b', label='Group 3 bar ellipticity', linestyle=':')
        ax2.tick_params(axis='both', which='both', labelsize=fs)
        ax2.set_ylabel(y2lab, fontsize=fs, c='k')
        ax2.legend(loc="upper right")
        #ax2.axis('equal')

    #plt.legend()
    #ax.axis('equal')
    #leg = ax.legend(loc="upper left")

    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_xlim(np.nanmin(snap_ages), np.nanmax(snap_ages))
    if plot_bar_ellipticity and plot_sigma_amp:
        title = model.replace("run","") + " sigma amplitude peaks and bar ellipticities evolution"
    elif plot_bar_ellipticity:
        title = model.replace("run","") + " bar ellipticities evolution"
    elif plot_sigma_amp:
        title = model.replace("run","") + " sigma amplitude peaks evolution"
    else:
        title = model.replace("run","") + " WTF evolution"
    ax.title.set_text(title)

    if save_file:
        if plot_bar_ellipticity and plot_sigma_amp:
            image_name = image_dir + model.replace("run","") + '_sigma_amp_bar_ellip.png'
        elif plot_bar_ellipticity:
            image_name = image_dir + model.replace("run","") + '_bar_ellip.png'
        elif plot_sigma_amp:
            image_name = image_dir + model.replace("run","") + '_sigma_amp.png'
        else:
            image_name = image_dir + model.replace("run","") + '_WTF.png'
        plt.savefig(image_name)
        print("Image saved to",image_name)
    else:
        print("Image saving is turned off.")
    if show_plot:
        plt.show()
    else:
        print("On-screen ploting is turned off.")

    return None


# Remove as unused ?
def plot_density_hist2d(x,y): # Learned from Stuart Andersson
    plt.hist2d(x,y,bins=(np.linspace(-1,1,100),
        np.linspace(-1,1,100)),cmap='cubehelix',norm=LogNorm())
    plt.show()

    return None


# Create PDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_to_pdf(output_pdf, square_figures, long_figures):
    with PdfPages(output_pdf) as pdf:
        # Create a master A4 figure
        master_fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches

        # Define positions for the plots (left, bottom, width, height)
        positions = [ # 3 square plots side by side and 2 long plots stacked
            (0.05, 0.75, 0.27, 0.18),  # Square 1
            (0.35, 0.75, 0.27, 0.18),  # Square 2
            (0.65, 0.75, 0.27, 0.18),  # Square 3
            (0.05, 0.48, 0.9, 0.24),  # Long 1
            (0.05, 0.24, 0.9, 0.24),  # Long 2
            #(0.05, 0.00, 0.9, 0.24),  # Long 3
            #(0.05, 0.00, 0.9, 0.2),  # Long 4
            #(0.05, 0.00, 0.9, 0.2),  # Long 5
        ]

        # Embed square plots
        for pos, fig in zip(positions[:3], square_figures):  # Ensure all 3 square figures are placed
            ax = master_fig.add_axes(pos)
            ax.imshow(fig.canvas.buffer_rgba(), aspect="auto")
            ax.axis("off")  # Remove axes

        # Embed long plots
        for pos, fig in zip(positions[3:], long_figures):
            ax = master_fig.add_axes(pos)
            ax.imshow(fig.canvas.buffer_rgba(), aspect="auto")
            ax.axis("off")

        # Save the master figure as a single PDF page
        pdf.savefig(master_fig, bbox_inches="tight")
        plt.close(master_fig)
