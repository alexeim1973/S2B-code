import numpy as np

# 3D array 
pos = [[14, 17, 12],  
       [15, 6, 27], 
       [23, 2, 54]] 

vel = [[140, 170, 120],  
       [-150, -60, -270], 
       [230, -2, -54]] 
    
# mean of the flattened array 
print("\nmean of arr, axis = None : ", np.mean(pos)) 
    
# mean along the axis = 0 
print("\nmean of arr, axis = 0 : ", np.mean(pos, axis = 0)) 
   
# mean along the axis = 1 
print("\nmean of arr, axis = 1 : ", np.mean(pos, axis = 1))
  
out_arr = np.arange(3)
print("\nout_arr : ", out_arr) 
print("mean of arr, axis = 1 : ", 
      np.mean(pos, axis = 1, out = out_arr))