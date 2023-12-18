# Import Library
import numpy as np 
import matplotlib.pyplot as plt

# Data Coordinates
x = np.arange(5, 10) 
y = np.arange(12, 17)
# PLot
plt.plot(x,y) 
# Add Title
plt.title("Matplotlib PLot NumPy Array") 
# Add Axes Labels
plt.xlabel("x axis") 
plt.ylabel("y axis") 
# Display
plt.show()
# Save
plt.savefig('images/test1.png')
#plt.savefig('tests/images/test1.pdf')

