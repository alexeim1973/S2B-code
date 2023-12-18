import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib import colors

# Generate sample 2D data (replace this with your own data)
np.random.seed(42)
x = np.random.uniform(0, 100, size=20)
y = np.random.uniform(0, 100, size=20)

#print("X:")
#print(x)

#print("Y:")
#print(y)

size = 20

m = [[1 for x in range(size)] for y in range(size)] 

for i in range(size):
    for j in range(size):
        if i > 8 and i < 11:
            if j > 1 and j < 18:
                #print(m[i][j])
                m[i][j] = 10
        elif i == 8 or i == 11:
            if j > 3 and j < 16:
                m[i][j] = 10

for x in m:
    print(x)

array = np.array(m)

# create discrete colormap
cmap = colors.ListedColormap(['blue', 'red'])
bounds = [0,10,20]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots()
ax.imshow(array, cmap=cmap, norm=norm, extent = [-size/2, size/2, -size/2, size/2 ],)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
ax.set_xticks(np.arange(-10, 11, 1));
ax.set_yticks(np.arange(-10, 11, 1));

for r in range(int(size/2)+1):
    circle = plt.Circle((0, 0), r, color='white', edgecolor='white', fill=False)
    plt.gca().add_patch(circle)
    print("r:", r)
    x = r + int(size/2)
    y = r + int(size/2)
    print("(x,y):", x, y, "m:", m[x][y])
    x = r + int(size/2)
    y = r + int(size/2) - 1
    print("(x,y):", x, y, "m:", m[x][y])
    x = r + int(size/2) - 1
    y = r + int(size/2)
    print("(x,y):", x, y, "m:", m[x][y])
    x = r + int(size/2) - 1
    y = r + int(size/2) - 1
    print("(x,y):", x, y, "m:", m[x][y])

#plt.show()

# Save
plt.savefig('images/bin-test1.png')
