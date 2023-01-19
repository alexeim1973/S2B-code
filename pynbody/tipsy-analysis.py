import yt
import matplotlib.pyplot as plt
import numpy as np

name = 'TipsyGalaxy'
ds = yt.load_sample(name)
# ds = yt.load_sample('run741CU')

print(ds.field_list)

ad = ds.all_data()
xcoord = ad['Gas', 'Coordinates'][:,0].v
ycoord = ad['Gas', 'Coordinates'][:,1].v
logT = np.log10(ad['Gas', 'Temperature'])
plt.scatter(xcoord, ycoord, c=logT, s=2*logT, marker='o', edgecolor='none', vmin=2, vmax=6)
plt.xlim(-20,20)
plt.ylim(-20,20)
cb = plt.colorbar()
cb.set_label('$\log_{10}$ Temperature')
plt.gcf().set_size_inches(15,10)
image_fname = name + '.png'
plt.savefig(image_fname)
plt.clf()