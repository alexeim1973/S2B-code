import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

data = np.random.rand(20, 20) * 20

# create discrete colormap
cmap = colors.ListedColormap(['red', 'blue'])
bounds = [0,10,20]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
ax.set_xticks(np.arange(-10, 10, 1));
ax.set_yticks(np.arange(-10, 10, 1));

plt.show()

# Save
plt.savefig('images/gridtest1.png')