import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

L = 10
x1, y1 = 1, 0
x2, y2 = 5, 6
n = 10

def f(x):
    global L, x1, x2, y1, y2
    a, c1, c2 = x
    eq1 = a*np.sinh((x2+c1)/a) - a*np.sinh((x1+c1)/a) - L
    eq2 = a*np.cosh((x1+c1)/a) + c2 -y1
    eq3 = a*np.cosh((x2+c1)/a) + c2 - y2
    return [eq1, eq2, eq3]



a, c1, c2 =  fsolve(f, (0.8, (x1+x2)/2, (y1+y2)/2), factor=0.1)

def g(p, i):
    global L, a, c1, c2, n
    global x1
    eq1 = a*np.sinh((p[0]+c1)/a) - a*np.sinh((x1+c1)/a) - i * L / (n - 1)
    eq2 = a*np.cosh((p[0]+c1)/a) + c2 - p[1]
    return [eq1, eq2]

t = np.linspace(x1, x2, 100)
c = a*np.cosh((t+c1)/a)+c2

print(a, c1, c2)
print(a*np.cosh((x1+c1)/a)+c2)
print(a*np.cosh((x2+c1)/a)+c2)
print(a*np.sinh((x2+c1)/a) - a*np.sinh((x1+c1)/a))

for i in range (1, n-1):
    x, y =  fsolve(g, (x1+i*(x2-x1)/(n-1), a*np.cosh((x1+i*(x2-x1)+c1)/((n-1)*a))+c2), args=(i))
    plt.scatter(x, y, color="crimson")

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.plot(t, c)
plt.grid(True)
plt.show()
