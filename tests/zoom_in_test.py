# Import Libraries
import numpy as np 
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pandas as pd
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

def ZoomIn(data,rad):
    mask = data[:,0] < rad
    print(len(data[:,0]) - sum(mask))
    data=data[mask]
    mask = data[:,0] > -1*rad
    print(len(data[:,0]) - sum(mask))
    data=data[mask]
    mask = data[:,1] < rad
    print(len(data[:,1]) - sum(mask))
    data=data[mask]
    mask = data[:,1] > -1*rad
    print(len(data[:,1]) - sum(mask))
    data=data[mask]
    return data

if __name__ == "__main__":

    data = np.array([[9, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, -9, 0]])
    #print(len(data[:,0]))
    rad = 8
    print(data)
    #data = ZoomIn(data,rad)
    print(np.transpose(data))

