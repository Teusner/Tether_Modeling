import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

L = 10
x1, y1 = 0, 3
x2, y2 = 4, 5

def f(x):
    L = 10
    x1, y1 = 0, 3
    x2, y2 = 4, 5
    a, c1, c2 = x
    eq1 = a*np.sinh((x2+c1)/a) - a*np.sinh((x1+c1)/a) - L
    eq2 = a*np.cosh((x1+c1)/a) + c2 -y1
    eq3 = a*np.cosh((x2+c1)/a) + c2 - y2
    return [eq1, eq2, eq3]

a, c1, c2 =  fsolve(f, (0.8, (x1+x2)/2, (y1+y2)/2), factor=0.1)

t = np.linspace(x1, x2, 100)
c = a*np.cosh((t+c1)/a)+c2

print(a, c1, c2)
print(a*np.cosh((x1+c1)/a)+c2)
print(a*np.cosh((x2+c1)/a)+c2)
print(a*np.sinh((x2+c1)/a) - a*np.sinh((x1+c1)/a))

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.plot(t, c)
plt.grid(True)
plt.show()
