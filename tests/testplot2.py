# Import Library
import numpy as np 
import matplotlib.pyplot as plt

def f(x, y):
        return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, colors='black')
plt.savefig('tests/images/test2.png')

plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()
plt.savefig('tests/images/test3.png')

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',cmap='RdGy')
plt.colorbar()
plt.savefig('tests/images/test4.png')