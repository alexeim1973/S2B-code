"""
Function to return the Gauss Hermite value of order n
for a given distribution of weights w.

@author: Steven Gough-Kelly
"""

# Import Libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pandas as pd
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

datafile = "GaiaChallenge/modelR1GaiaChallenge.csv"

def Gauss_Hermite(w, n):
    """
    Return the Gauss Hermite function of order n, weights w
    Gerhard MNRAS (1993) 265, 213-230
    Equations 3.1 - 3.7
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

def LoadData():
    start_time = datetime.now() 
    df = pd.read_csv(datafile,header=None)
    time_elapsed = datetime.now() - start_time 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    df.columns = ["x", "y", "z", "vx","vy","vz"]
    return df

def mat_rotate_z(x,y,z, angle):
        angle = np.radians(angle)
        mat = np.matrix([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle) , 0],
                         [0            , 0             , 1]])
        return np.array(mat.dot([x, y, z]))

def mat_rotate_y(x, y, z, angle):
        angle = np.radians(angle)
        mat = np.matrix([[np.cos(angle),   0,   np.sin(angle)],
                        [0,                1,               0],
                        [-np.sin(angle),   0,   np.cos(angle)]])
        return np.array(mat.dot([x, y, z]))

def mat_rotate_x(x, y, z, angle):
        angle = np.radians(angle)
        mat = np.matrix([[1,             0,              0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle),  np.cos(angle)]])
        return np.array(mat.dot([x, y, z]))

if __name__ == "__main__":

    """
    Example of using these functions with binned_statistic. These values are not
    representative of true distributions of positions and velocity just to debug.
    """

    data = LoadData()
    x = data['x'].values
    y = data['y'].values
    z = data['z'].values
    vx = data['vx'].values
    vy = data['vy'].values
    vz = data['vz'].values

    bins = 10
    
    xlim = x.min(), x.max()
    ylim = y.min(), y.max()

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    # fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharey=True)
    # fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharey=True, figsize=(9, 4))

    cmap = 'seismic'

    '''
    hb = ax0.hexbin(x, y, gridsize=50, bins='log', cmap=cmap)
    ax0.set(xlim=xlim, ylim=ylim)
    ax0.set_aspect(1)
    ax0.set_title("R1 face-on")
    ax0.set_xlabel("kpc")
    ax0.set_ylabel("kpc")
    # ax0.contour(x, y, z, 10, colors='white')
    # cb = fig.colorbar(hb, ax=ax1, label='log10(N)')
    '''

    # Rotate to 90 around X axis for edge-on view
    datart = mat_rotate_x(x, y, z, 90)
    x = datart[0,:]
    y = datart[1,:]
    z = datart[2,:]
    
    xlim = x.min(), x.max()
    ylim = y.min(), y.max()
    
    plt.hist2d(x, y, gridsize=50, bins='log', cmap=cmap)
    plt(xlim=xlim, ylim=ylim)
    axes=plt.gca()
    axes.set_aspect(1)
    plt.title("R1 edge-on")
    plt.xlabel("kpc")
    plt.ylabel("kpc")
    cb = plt.colorbar()
    cb.set_label('log10 num density')
    plt.savefig('GaiaChallenge/images/R1-hist2d-test1.png')
    plt.clf()
    

    # Rotate to 90 around Y axis for side-on view
    datart = mat_rotate_y(x, y, z, 90)
    x = datart[0,:]
    y = datart[1,:]
    z = datart[2,:]

    xlim = x.min(), x.max()
    ylim = y.min(), y.max()

    plt.hist2d(x, y, gridsize=50, bins='log', cmap=cmap)
    plt(xlim=xlim, ylim=ylim)
    axes=plt.gca()
    axes.set_aspect(1)
    plt.title("R1 side-on")
    plt.xlabel("kpc")
    plt.ylabel("kpc")
    plt.savefig('R1-hist2d-test1.png')
    cb = plt.colorbar()
    cb.set_label('log10 num density')
    plt.savefig('GaiaChallenge/images/R1-hist2d-test2.png')
    plt.clf()

    #print(len(x))
    #print(len(y))
    #print(len(vz))
    '''
    stat,edges,binnum = stats.binned_statistic(x, vz,
                                   statistic=lambda bin_values: \
                                   GaussHermiteMoment(bin_values, 4),
                                   bins=bins)
    
    print()
    print('stat = ' + str(stat))
    print('edges = ' + str(edges))
    print('binnum = ' + str(binnum))
    
    stat2d,xedges,yedges,binnum2d = stats.binned_statistic_2d(x, y ,vz,
                                  statistic=lambda bin_values: \
                                  GaussHermiteMoment(bin_values, 4),
                                  bins=bins)
    
    print()
    print('sta2d = ' + str(stat2d))
    print('xedges = ' + str(xedges))
    print('yedges = ' + str(yedges))
    print('binnum2d = ' + str(binnum2d))
    '''

    #print(len(stat),len(edges),len(binnum))
    #print()
    #print(len(stat2d),len(xedges),len(yedges),len(binnum2d))
    #print(stat2d,xedges,yedges,binnum2d)