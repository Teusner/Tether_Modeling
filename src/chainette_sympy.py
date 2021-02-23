import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Parameters
    L, x1, y1, xn, yn = sp.symbols("L x1 y1 xn yn")
    L = 10
    x1, y1 = 0, 3
    xn, yn = 4, 5

    # Variables
    c1, c2, c3 = sp.symbols("c1 c2 c3")
    
    eq1 = sp.Eq(c1 * (sp.sinh((xn+c2)/c1) - sp.sinh((x1+c2)/c1)), L)
    eq2 = sp.Eq(c1 * sp.cosh((x1+c2)/c1) + c3, y1)
    eq3 = sp.Eq(c1 * sp.cosh((xn+c2)/c1) + c3, yn)

    sp.pprint(eq1)
    sp.pprint(eq2)
    sp.pprint(eq3)

    # res, = sp.nonlinsolve((eq1, eq2, eq3), (c1, c2, c3))

    # sp.pprint(res)

    L = 10
    x1, y1 = 0, 3
    xn, yn = 4, 5
    c1, c2, c3 =  3.0, 5.0, - 3.0 * np.cosh(5/3) + 3 # -c1*np.cosh(c2/c1+x1/c1) + y1

    t = np.linspace(x1, xn, 100)
    c = c1*np.cosh((t+c1)/c1)+c3

    print(c1, c2, c3)
    print(c1*np.cosh((x1+c2)/c1)+c3)
    print(c1*np.cosh((xn+c2)/c1)+c3)
    print(c1*np.sinh((xn+c2)/c1) - c1*np.sinh((x1+c2)/c1))

    plt.scatter(x1, y1)
    plt.scatter(xn, yn)
    plt.plot(t, c)
    plt.grid(True)
    plt.show()
