# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Creating dataset
x = np.random.normal(size = 500000)
y = x * 3 + 4 * np.random.normal(size = 500000)

fig, ax = plt.subplots(figsize =(10, 7))
# Creating plot
plt.title("Using matplotlib hexbin function")
plt.hexbin(x, y, bins = 50)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# show plot
plt.tight_layout()
plt.savefig('tests/images/hexbin-test1.png')
