import numpy as np
from datetime import datetime 

start_time = datetime.now() 

# INSERT YOUR CODE 
dt = [('x','f8'),('y','f8'),('z','f8'),('vx','f8'),('vy','f8'),('vz','f8')]
# data = np.genfromtxt('GaiaChallenge/modelR1GaiaChallenge.csv',dtype=dt,delimiter=',')
data = np.genfromtxt('GaiaChallenge/modelR1GaiaChallenge',dtype=dt)

time_elapsed = datetime.now() - start_time 

print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

print(data[0][0])
