# import modules
from matplotlib.pylab import *
import pynbody as pb
import numpy as np
from scipy import stats as st

# Define global variables
filename = "TBA" # Add a path to your file(s)
paramfile = "TBA" # Add a path to your pynbody parameter file(s)
nuclear_bar = False # For TNG50

# Align the bar during the data load
align = True

# If the model is a high resolution
model_high_res = True
model_high_res = False # TNG50

if model_high_res:
    bin_width = 0.1 # kpc - for Amp/Phase calculations
    bin_arc = 10 # degrees
    if nuclear_bar:
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
    xlim, ylim = 7., 7. # kpc - primary bar plus a bit more


# Alligns the bar to x-axis, using inertia tensor
def bar_align(galaxy, nuclear_bar = False, log = False):

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
            print('* Aligning the snapshot...')

    if nuclear_bar:
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
    
    if log:
        print("From function pbload of module my_functions.py.")
        print("Data file:", filename)
        print("Param file:", paramname)
        print("Nuclear bar:", nuclear)
        print("Align bar to the X-acis:", align)
        print("Verbose logging:", log)

    # Loading data from file
    if log:
        print('* Loading data from file', filename, '...')
    
    if '::' in filename:
        filename, species = filename.split('::')
        sim = pb.load(filename, paramname=paramname)
        sim = getattr(sim, species)
    else:
        sim = pb.load(filename, paramname=paramname)

    sim.physical_units()

    if log:
        print('* Rotating the snapshot face on...')
    
    #Rotate the simulation so that we see the stars face on
    pb.analysis.angmom.faceon(sim.s)

    # Centering the snapshot
    if log:
        print('* Centering the snapshot...')

    #Centre the stars using a hybrid method - potential minimum and shrinking sphere
    pb.analysis.halo.center(sim.s, mode='pot')
    
    # Aligning the data
    if align:
        bar_align(sim, nuclear_bar, log)

    return sim


# This function extracts a sub-list of limited dimentions m by m 
# from original list n by n,
# centered as nested squares.
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


# This function calculates sums on rows and collumns for a given 2D list.
def sum_columns_and_rows(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(num_rows)) for j in range(num_cols)]

    return row_sums, col_sums



# This function calculates a bar ellipticity using quadrupole method
def ellipticity_quadrupole(unit,bins,stats):
    q_x = 0
    q_y = 0

    for R in range (1, int(bins/2)+1):

        sub_stats = extract_sublist(stats, R)

        row_sums, col_sums = sum_columns_and_rows(sub_stats)

        r = unit*R

        q_y = q_y + row_sums[R]*(r**2)
        q_x = q_x + col_sums[R]*(r**2)

    # SQRT function here is used from matplotlib.pylab, 
    # one can replace it with SQRT function from numpy
    e = 1 - sqrt(q_y / q_x)

    return(round(e,2))


if __name__ == '__main__':

    sim = pbload(filename, paramfile, nuclear_bar, align, log)

    sim.physical_units()

    #Extract phase space data for the model for stars in the group
    z, x, y = sim.star['z'], sim.star['x'], sim.star['y']
    
    # Number density statistics face-on for stellar population by age group
    d_stat2d,d_xedges,d_yedges,df_binnum2d = st.binned_statistic_2d(x, y, z,
                                    statistic = 'count',
                                    range = [[-xlim,xlim],[-ylim,ylim]],
                                    bins = bins)

    unit = 2*xlim/bins

    if log:
        print("Unit in kpc:", round(unit,2))

    # Arrays for ellipticies and radii
    e_list = []
    r_list = []

    # calculate ellipticity per radius at bin edge, 
    # the d_stst2d is already cut by xlim,ylim
    for radius in range(1,int(bins/2)+1):
        sub_list = extract_sublist(d_stat2d, radius)
        e = ellipticity_quadrupole(unit,2*radius,sub_list)
        e_list.append(e)
        r_list.append(round(radius*unit,2))
