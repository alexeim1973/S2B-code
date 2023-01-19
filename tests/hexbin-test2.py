# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Creating dataset
x = np.random.normal(size = 500000)
y = x * 3 + 4 * np.random.normal(size = 500000)
z = y * 3 + 4 * np.random.normal(size = 500000)

# Creating bins
x_min = np.min(x)
x_max = np.max(x)

y_min = np.min(y)
y_max = np.max(y)

x_bins = np.linspace(x_min, x_max, 50)
y_bins = np.linspace(y_min, y_max, 20)

fig, ax = plt.subplots(figsize =(10, 7))
# Creating plot
plt.hist2d(x, y, bins =[x_bins, y_bins], cmap = plt.cm.nipy_spectral)
plt.title("Changing the color scale and adding color bar")

# Adding color bar
plt.colorbar()

ax.set_xlabel('X-axis')
ax.set_ylabel('X-axis')

# show plot
plt.tight_layout()
plt.savefig('tests/images/hexbin-test.png')
