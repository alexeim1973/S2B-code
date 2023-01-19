"""
Plot velocity distriution and GH moment 3 and 4

@author: Alexei Monastyrnyi
"""

# Import Libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy
import math
from scipy import stats as st
from collections import namedtuple
import pandas as pd
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pynbody as pb

# Define global variables here
datafile = "GaiaChallenge/modelR1GaiaChallenge.csv"
data_fname1 = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.00500.gz'
data_fname2 = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.01000.gz'
param_fname = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.param'
density_image = 'pynbody/images/plot-density.png'
moment3_image = 'pynbody/images/plot-moment3.png'
moment4_image = 'pynbody/images/plot-moment4.png'
density_title = 'R1 edge-on - log10 number density'
moment3_title = 'R1 edge-on - h3 moment'
moment4_title = 'R1 edge-on - h4 moment'
density_cbar_label = 'Log10 number density'
moment3_cbar_label = 'Velocity stat2d'
moment4_cbar_label = 'Velocity stat2d'
bins = 75
xlim, ylim = 8, 3
#xylim = [[-xlim, xlim], [-ylim, ylim]]
cmap = 'seismic'
density = {}
t_unit = 'Myr'
d_unit = 'kpc'
v_unit = 'km s^-1'
ro_unit = 'g cm^-3'

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

def mat_rotate_z(x,y,z, angle):
    angle = np.radians(angle)
    mat = np.matrix([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle) , 0],
                     [0            , 0             , 1]])
    return np.array(mat.dot([x, y, z])).transpose() # Transpose is added by Alexei

def mat_rotate_y(x, y, z, angle):
    angle = np.radians(angle)
    mat = np.matrix([[np.cos(angle),   0,   np.sin(angle)],
                    [0,                1,               0],
                    [-np.sin(angle),   0,   np.cos(angle)]])
    return np.array(mat.dot([x, y, z])).transpose() # Transpose is added by Alexei

def mat_rotate_x(x, y, z, angle):
    angle = np.radians(angle)
    mat = np.matrix([[1,             0,              0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])
    return np.array(mat.dot([x, y, z])).transpose() # Transpose is added by Alexei

def LoadData():
    print('Loading data from file...')
    start_time = datetime.now() 
    df = pd.read_csv(datafile,header=None)
    print('Done!')
    time_elapsed = datetime.now() - start_time
    print('Elapsed time (hh:mm:ss.ms) {}'.format(time_elapsed))
    print()
    df.columns = ["x", "y", "z", "vx","vy","vz"]
    return df       

def pbload(filename, paramname=None):
    if '::' in filename:
        filename, species = filename.split('::')
        sim = pb.load(filename, paramname=paramname)
        sim = getattr(sim, species)
    else:
        sim = pb.load(filename, paramname=paramname)
    return sim

def RenderDensity(plt_title,cbar_label,image_fname):
    print('Rendering number density stats...')
    start_time = datetime.now() 
    stat2d,xedges,yedges,binnum2d = st.binned_statistic_2d(x, y, vz,
                              statistic = 'count',
                              #range = xylim,
                              bins = bins)
    xmid = np.round((xedges[1:] + xedges[:-1]) / 2,1)
    ymid = np.round((yedges[1:] + yedges[:-1]) / 2,1) 
    z = np.flip(stat2d.transpose(), axis=0)
    z = np.where(z > 0, z, 0.5)
    z = np.log10(z)
    density['x'] = xmid
    density['y'] = ymid
    density['z'] = z
    print('Done!')
    time_elapsed = datetime.now() - start_time
    print('Elapsed time (hh:mm:ss.ms) {}'.format(time_elapsed))
    print()
    plot_data(xmid, ymid, z, plt_title, cbar_label, image_fname)
        
def RenderMoment(m,plt_title,cbar_label,image_fname):
    print('Rendering h' + str(m) + ' moment stats...')
    start_time = datetime.now() 
    stat2d,xedges,yedges,binnum2d = st.binned_statistic_2d(x, y ,vz,
                              statistic=lambda bin_values: \
                              GaussHermiteMoment(bin_values, m),
                              #range = xylim,
                              bins = bins)
    xmid = np.round((xedges[1:] + xedges[:-1]) / 2,1)
    ymid = np.round((yedges[1:] + yedges[:-1]) / 2,1)
    z = np.flip(stat2d.transpose(), axis=0)
    print('Done!')
    time_elapsed = datetime.now() - start_time
    print('Elapsed time (hh:mm:ss.ms) {}'.format(time_elapsed))
    print()
    plot_data(xmid, ymid, z, plt_title, cbar_label, image_fname)    
        
def plot_data(xmid, ymid, z, plt_title, cbar_label, image_fname):
    plt.pcolormesh(xmid, ymid, z, cmap=cmap)
    plt.title(plt_title)  
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    axes = plt.gca()
    axes.set_aspect(1)
    axes.contour(density['x'], density['y'], density['z'], colors='k')
    axes.set_xlabel('kpc')
    axes.set_ylabel('kpc')
    axes.set_xlim(-xlim, xlim)
    axes.set_ylim(-ylim, ylim)
    plt.savefig(image_fname)
    print('Image ' + plt_title + ' is saved to the file ' + image_fname)
    print()
    plt.clf()

if __name__ == "__main__":

    s = pbload(data_fname1,param_fname)
    
    x = s.s['pos'][:,0].in_units(d_unit)
    y = s.s['pos'][:,1].in_units(d_unit)
    z = s.s['pos'][:,2].in_units(d_unit)
    vx = s.s['vel'][:,0].in_units(v_unit)
    vy = s.s['vel'][:,1].in_units(v_unit)
    vz = s.s['vel'][:,2].in_units(v_unit)

    print(x, y, z)
    print()
    print(vx, vy, vz)
    print()

    # Rotate to 90 around X axis for edge-on view
    pos_data = mat_rotate_x(x, y, z, 90)
    x = pos_data[:,0]
    y = pos_data[:,1]
    z = pos_data[:,2]

    v_data = mat_rotate_x(vx, vy, vz, 90)
    vx = v_data[:,0]
    vy = v_data[:,1]
    vz = v_data[:,2]

    RenderDensity(density_title,density_cbar_label,density_image)
    RenderMoment(3,moment3_title,moment3_cbar_label,moment3_image)
    RenderMoment(4,moment4_title,moment4_cbar_label,moment4_image)
