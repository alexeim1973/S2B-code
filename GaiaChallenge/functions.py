"""
    Return the Gauss Hermite function of order n, weights w
    Gerhard MNRAS (1993) 265, 213-230
    Equations 3.1 - 3.7
"""

import numpy as np
import scipy
from scipy import stats


def Gauss_Hermite(w, n):
    w = np.array(w)
    #orders = np.arange(n+1)[::-1] #generates n-0 Stefan, removed, redundant
    p = scipy.special.hermite(n, monic=False) #hermite poly1d obj @UndefinedVariable
    norm = np.sqrt((2**(n+1))*np.pi*np.math.factorial(n)) # N_n Eqn 3.1
    return (p(w)/norm) * np.exp( -0.5 * w * w )


def GaussHermiteMoment(v, n):
    v = v[np.isfinite(v)] # remove nans&inf
    if len(v) <= 1:         # Added by Stefan
        return np.nan
    v_dash = (v - np.mean(v))/np.std(v) # center on 0, norm width to 1sig
    hn = np.sum(Gauss_Hermite(v_dash, n))
    return np.sqrt(4*np.pi) * hn / len(v)


def testCase():
    x = np.random.normal(0,100,100)
    y = np.random.normal(0,100,100)
    vz = np.random.normal(0,100,100)
    bins = 10
    stat,edges,binnum = stats.binned_statistic(x, vz,
                                   statistic=lambda bin_values: \
                                   GaussHermiteMoment(bin_values, 4),
                                   bins=bins)

    print('stat = ' + str(stat))
    print('edges = ' + str(edges))
    print('binnum = ' + str(binnum))

    stat2d,xedges,yedges,binnum2d = stats.binned_statistic_2d(x, y ,vz,
                                statistic=lambda bin_values: \
                                GaussHermiteMoment(bin_values, 4),
                                bins=bins)

    print('sta2d = ' + str(stat2d))
    print('xedges = ' + str(xedges))
    print('yedges = ' + str(yedges))
    print('binnum2d = ' + str(binnum2d))

# R1GaiaChallenge is face-on. Stefan needs to rotate it and Steven provided these:

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