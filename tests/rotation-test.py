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

    data = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],[0, 1, 0]])

    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    plt.scatter(x, y, c='red', s=20, alpha=0.3,
                   cmap='viridis')
    plt.savefig('tests/images/rotation-test1.png')

    data90x = mat_rotate_x(x, y, z, 90)
    data180x = mat_rotate_x(x, y, z, 180)
    data270x = mat_rotate_x(x, y, z, 270)
    data360x = mat_rotate_x(x, y, z, 360)
    
    print('X rotation 0, 90, 180, 270, 360')
    print(data)
    print(data90x.round())
    print(data180x.round())
    print(data270x.round())
    print(data360x.round())
    print()
    
    data90y = mat_rotate_y(x, y, z, 90)
    data180y = mat_rotate_y(x, y, z, 180)
    data270y = mat_rotate_y(x, y, z, 270)
    data360y = mat_rotate_y(x, y, z, 360)

    print('Y rotation 0, 90, 180, 270, 360')
    print(data)
    print(data90y.round())
    print(data180y.round())
    print(data270y.round())
    print(data360y.round())
    print()
    
    data90z = mat_rotate_z(x, y, z, 90)
    data180z = mat_rotate_z(x, y, z, 180)
    data270z = mat_rotate_z(x, y, z, 270)
    data360z = mat_rotate_z(x, y, z, 360)

    print('Z rotation 0, 90, 180, 270, 360')
    print(data)
    print(data90z.round())
    print(data180z.round())
    print(data270z.round())
    print(data360z.round())

    x = data90x[:,0]
    y = data90x[:,1]
    z = data90x[:,2]

    plt.scatter(x, y, c='blue', s=20, alpha=0.3,
                   cmap='viridis')
    plt.savefig('tests/images/rotation-test2.png')