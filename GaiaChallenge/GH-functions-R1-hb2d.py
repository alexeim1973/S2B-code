"""
Function to return the Gauss Hermite value of order n
for a given distribution of weights w.

@author: Steven Gough-Kelly
"""

# Import Libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy
from scipy import stats as st
import pandas as pd
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def LoadData():
    start_time = datetime.now() 
    df = pd.read_csv(datafile,header=None)
    time_elapsed = datetime.now() - start_time 
    print('Data load elapsed time (hh:mm:ss.ms) {}'.format(time_elapsed))
    df.columns = ["x", "y", "z", "vx","vy","vz"]
    return df

def RenderDensity(x,y,vz,title,image_fname):
    stat2d,xedges,yedges,_ = st.binned_statistic_2d(x, y, vz,
                              statistic='count',
                              range = lim2D,
                              bins=bins)
    xmid = np.round((xedges[1:] + xedges[:-1]) / 2,1)
    ymid = np.round((yedges[1:] + yedges[:-1]) / 2,1) 
    z = np.flip(stat2d.transpose(), axis=0)
    z = np.where(z > 0, z, 0.5)
    z = np.log10(z)
    density = {}
    density['x'] = xmid
    density['y'] = ymid
    density['density'] = z
    plot_data(xmid, ymid, z, title, image_fname)

def RenderMoment(x,y,vz,m,p_title,image_fname):
    z,xedges,yedges, _ = st.binned_statistic_2d(x, y ,vz,
                              statistic=lambda bin_values: \
                              GaussHermiteMoment(bin_values, m),
                              range = lim2D,
                              bins = bins)
    xmid = np.round((xedges[1:] + xedges[:-1]) / 2,1)
    ymid = np.round((yedges[1:] + yedges[:-1]) / 2,1)
    z = np.flip(z.transpose(), axis=0)
    plot_data(xmid, ymid, z, p_title, image_fname)

def plot_hb2d(x,y,p_title,img_f):
        #xlim = x.min(), x.max()
        #ylim = y.min(), y.max()
        hb = plt.hexbin(x, y, gridsize=50, bins='log', cmap=cmap)
        plt.xlim(xlim)
        plt.ylim(ylim)
        axes = plt.gca()
        axes.set_aspect(1)
        plt.title(p_title)
        plt.xlabel("kpc")
        plt.ylabel("kpc")
        # cb = fig.colorbar(hb, ax=plt, label='log10(N)')
        cb = plt.colorbar()
        cb.set_label('log10 num density')
        plt.savefig(img_f)
        plt.clf()

def plot_imshow(x,y,z,p_title,img_f):
        plt.imshow(z, cmap=cmap, vmin=z.min(), vmax=z.max(),
              extent=[x.min(), x.max(), y.min(), y.max()],
              interpolation='nearest', origin='lower', aspect='auto')
        plt.xlim(xlim)
        plt.ylim(ylim)
        axes = plt.gca()
        axes.set_aspect(1)
        plt.title(p_title)
        plt.xlabel("kpc")
        plt.ylabel("kpc")
        cb = plt.colorbar()
        cb.set_label('log10 num density')
        plt.savefig(img_f)
        plt.clf()
        

def plot_data(xmid, ymid, z, p_title, image_fname):
    plt.pcolor(xmid, ymid, z, cmap=cmap)
    plt.title(p_title)
    #fig.colorbar(c, ax=ax)
    # con = ax.contour(density['x'], density['y'], density['density'],
    #                          colors='k')
    ax = plt.gca()
    ax.set_aspect(1)
    ax.set_xlabel('kpc')
    ax.set_ylabel('kpc')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cb = plt.colorbar()
    cb.set_label('log10 num density')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(image_fname)
    plt.clf()

def ZoomIn(data,rad):
    mask = data[:,0] < rad
    data = data[mask]
    mask = data[:,0] > -1*rad
    data = data[mask]
    mask = data[:,1] < rad
    data = data[mask]
    mask = data[:,1] > -1*rad
    data = data[mask]
    return data

# Define global variables here
datafile = "GaiaChallenge/modelR1GaiaChallenge.csv"
rad = 8 #kpc
xlim = -15, 15
ylim = -15, 15
lim2D = [xlim, ylim]
bins = 75

if __name__ == "__main__":

    fulldata = LoadData()
   
    x = fulldata['x'].values
    y = fulldata['y'].values
    z = fulldata['z'].values
    vx = fulldata['vx'].values
    vy = fulldata['vy'].values
    vz = fulldata['vz'].values

    '''
    data = np.array([x, y, z])
    data = np.transpose(data)
    data = ZoomIn(data,rad)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    '''

    cmap = 'seismic'
   
    # fig, (plt, ax2) = plt.subplots(ncols=2, sharey=True)
    # fig, (ax0, plt, ax2) = plt.subplots(ncols=3, sharey=True)
    # fig, (ax0, plt, ax2) = plt.subplots(ncols=3, sharey=True, figsize=(9, 4))

    p_title = "Face-on"
    img_f = 'GaiaChallenge/images/R1-hb2d-' + p_title + '.png'
    plot_hb2d(x,y,p_title,img_f)

    # Rotate to 90 around X axis for edge-on view
    data = mat_rotate_x(x, y, z, 90)
    data = np.transpose(data)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    
    #p_title = "Edge-on"
    #img_f = 'images/R1-hb2d-' + p_title + '.png'
    #plot_hb2d(x,y,p_title,img_f)
    #img_f = 'images/R1-imshow-' + p_title + '.png'
    #plot_imshow(x,y,z,p_title,img_f)
    p_title = 'edge-on-num-density'
    img_f = 'GaiaChallenge/images/R1-hb2d-' + p_title + '.png'
    RenderDensity(x,y,vz,p_title,img_f)

    p_title = 'h'+str(3)+'-moment'
    img_f = 'GaiaChallenge/images/R1-hb2d-' + p_title + '.png'
    RenderMoment(x,y,vz,3,p_title,img_f)

    p_title = 'h'+str(4)+'-moment'
    img_f = 'GaiaChallenge/images/R1-hb2d-' + p_title + '.png'
    RenderMoment(x,y,vz,4,p_title,img_f)

    # Rotate to 90 around Y axis for side-on view
    x = fulldata['x'].values
    y = fulldata['y'].values
    z = fulldata['z'].values
    vx = fulldata['vx'].values
    vy = fulldata['vy'].values
    vz = fulldata['vz'].values

    data = mat_rotate_y(x, y, z, 90)
    data = np.transpose(data)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    p_title = "Side-on"
    img_f = 'GaiaChallenge/images/R1-hb2d-'+ p_title
    plot_hb2d(x,y,p_title,img_f)

    #print(len(x))
    #print(len(y))
    #print(len(vz))
