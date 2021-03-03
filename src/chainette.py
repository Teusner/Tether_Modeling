import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def f(x):
    global L, x1, x2, y1, y2
    a, c1, c2 = x
    eq1 = a*np.sinh((x2+c1)/a) - a*np.sinh((x1+c1)/a) - L
    eq2 = a*np.cosh((x1+c1)/a) + c2 -y1
    eq3 = a*np.cosh((x2+c1)/a) + c2 - y2
    return [eq1, eq2, eq3]

def g(p, i):
    global L, a, c1, c2, n
    global x1
    eq1 = a*np.sinh((p[0]+c1)/a) - a*np.sinh((x1+c1)/a) - i * L / (n - 1)
    eq2 = a*np.cosh((p[0]+c1)/a) + c2 - p[1]
    return [eq1, eq2]


if __name__ == "__main__":
    # Catenary parameters
    L = 25
    x1, y1 = 11, 2
    x2, y2 = 18, -2
    n = 25

    # Getting catenary parameters
    a, c1, c2 =  fsolve(f, (1, -(x1+x2)/2, (y1+y2)/2), factor=0.1)
    print("a: {}, c1: {}, c2: {}".format(a, c1, c2))

    # Plotting the catenary curve
    t = np.linspace(x1, x2, 100)
    c = a*np.cosh((t+c1)/a)+c2

    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.plot(t, c)

    # Initializing each node
    for i in range (1, n-1):
        x, y =  fsolve(g, (x1+i*(x2-x1)/(n-1), a*np.cosh((x1+i*(x2-x1)+c1)/((n-1)*a))+c2), args=(i))
        plt.scatter(x, y, color="crimson")

    plt.axis("equal")
    plt.grid(True)
    plt.show()
