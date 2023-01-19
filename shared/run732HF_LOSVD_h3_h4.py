from xml.etree.ElementInclude import include
import scipy
import pynbody as pb
from scipy import stats as st
from matplotlib.colors import LogNorm
from matplotlib.pylab import *

def pbload(filename, paramname=None):
    print('Loading data from file', filename, '...')
    if '::' in filename:
        filename, species = filename.split('::')
        sim = pb.load(filename, paramname=paramname)
        sim = getattr(sim, species)
    else:
        sim = pb.load(filename, paramname=paramname)
    print('Done!')
    return sim

def Gauss_Hermite(w, n):
    """
    Return the Gauss Hermite function of order n, weights w
    Gerhard MNRAS (1993) 265, 213-230
    Equations 3.1 - 3.7
    @author: Steven Gough-Kelly
    """
    w = np.array(w)
    p = scipy.special.hermite(n, monic=False) #hermite poly1d obj
    norm = np.sqrt((2**(n+1))*np.pi*np.math.factorial(n)) # N_n Eqn 3.1
    return (p(w)/norm) * np.exp( -0.5 * w * w )

def GaussHermiteMoment(v, n):
    v = v[np.isfinite(v)] # remove nans&inf
    if len(v) <= 1: # Added SL speed+error catch
        return np.nan
    v_dash = (v - np.mean(v))/np.std(v) # center on 0, norm width to 1sig
    hn = np.sum(Gauss_Hermite(v_dash, n))
    return np.sqrt(4*np.pi) * hn / len(v)

def bar_align(galaxy, rbar, barfrac = 0.5, zlim=0.5, log=False):
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
    if np.isnan(rbar):
        if log:
            print('* Bar undefined, using 1 kpc *')
        rbar = 1.0
    elif rbar*barfrac < 1.:
        rbar = 1
        if log:
            print('* Short Bar, using 1 kpc *')
    else:
        rbar = rbar*barfrac
        if log:
            print('* Bar defined, aligning to {} kpc *'.format(rbar))

    if log:
        print('* Realigning bar using |z| < {} *'.format(zlim))

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
        print('* Bar realigned by {} degrees*'.format(r_angle))

    return None

def load_data_lst(fname_lst):
    s_lst = []
    for df in fname_lst:
        s = pbload(df,param_fname)
        pb.analysis.angmom.faceon(s)
        bar_align(s,3.,barfrac=1.,zlim=0.5,log=True)
        s_lst.append(s)
    return(s_lst)

def load_data(fname):
    s = pbload(fname,param_fname)
    pb.analysis.angmom.faceon(s)
    bar_align(s,3.,barfrac=1.,zlim=0.5,log=True)
    return(s)

def list_snaps(base_dir):
    import os
    # list to store files
    lst = []
    # Iterate directory
    for path in os.listdir(base_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(base_dir, path)) and '.gz' in path:
            lst.append(path)
    return(sorted(lst))

if __name__ == '__main__':

    base_dir = '/home/ubuntu/projects/S2B/models/run732HF/'
    data_fname1 = base_dir + 'run732HF.01200'
    param_fname = base_dir + 'run732HF.param'

    snap_lst = list_snaps(base_dir)
    # print(snap_lst)

    xlim, ylim = 1, 1 #kpc
    bins = 50
    cmap = 'seismic'
    plt_title = 'Face-on view - number density'
    cbar_label = 'Log10 number density'

    snap_cnt = len(snap_lst)

    # make the figure and sub plots
    fig,axes = plt.subplots(1,snap_cnt,figsize=(14,5))

    i = 0
    for snap in snap_lst:
    
        s = load_data(base_dir + snap)

        age = round(s.properties['time'].in_units('Gyr'),2)
     
        stat2d,xedges,yedges,binnum2d = st.binned_statistic_2d(s.star['x'], s.star['y'], s.star['z'],
                                statistic = 'count',
                                range = [[-xlim,xlim],[-ylim,ylim]],
                                bins = bins)
    
        image = axes[i].imshow(stat2d.T, 
                    origin = 'lower',
                    extent = [-xlim, xlim, -ylim, ylim ],
                    norm = LogNorm(),
                    cmap=cmap)

        axes[i].title.set_text(str(age) + ' Gyr')

        xcent = (xedges[1:] + xedges[:-1]) / 2
        ycent = (yedges[1:] + yedges[:-1]) / 2
        axes[i].contour(xcent, ycent, np.log10(stat2d.T), colors='k')
        i= i + 1
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
    cbar = fig.colorbar(image, cax = cbar_ax)
    cbar.set_label(cbar_label)
    fig.suptitle(plt_title)
    plt.setp(axes[:], xlabel='x [kpc]')
    plt.setp(axes[0], ylabel='y [kpc]')
    plt.savefig('inner_bar_dev.png')
    # plt.savefig(base_dir + 'images/' + 'inner_bar_dev.png')
    # plt.show()

