import numpy as np
from scipy.optimize import fsolve

def catenary(p, p1, pn, length):
    rmax = np.linalg.norm(pn[:2] - p1[:2])
    eq1 = p[0]*np.sinh((rmax+p[1])/p[0]) - p[0]*np.sinh((p[1])/p[0]) - length
    eq2 = p[0]*np.cosh((0+p[1])/p[0]) + p[2] - p1[2, 0]
    eq3 = p[0]*np.cosh((rmax+p[1])/p[0]) + p[2] - pn[2, 0]
    return [eq1, eq2, eq3]

def get_catenary_coefficients(p1, pn, length):
    return fsolve(catenary, (1., (p1[0, 0]-pn[0, 0])/2, (p1[2, 0]+pn[2, 0])/2), args=(p1, pn, length))

def initial_position(p, p1, pn, L, n, i, initial_parameters):
    eq1 = initial_parameters[0]*np.sinh((p[0]+initial_parameters[1])/initial_parameters[0]) - initial_parameters[0]*np.sinh((initial_parameters[1])/initial_parameters[0]) - i * L / n
    eq2 = initial_parameters[0]*np.cosh((p[0]+initial_parameters[1])/initial_parameters[0]) + initial_parameters[2] - p[1]
    eq3 = p1[3, 0] + i * (pn[3, 0] - p1[3, 0]) / (n - 1) - p[2]
    return [eq1, eq2, eq3]

def get_initial_position(p1, pn, L, n, i, initial_parameters):
    rmax = np.linalg.norm(pn[:2] - p1[:2])
    state = fsolve(initial_position, (rmax / 2, pn[1, 0]-pn[0, 0], pn[3, 0]-p1[3, 0]), args=(p1, pn, L, n, i, initial_parameters)).reshape(3, 1)
    theta = np.arctan2(pn[1] - p1[1], pn[0] - p1[0])
    x = np.cos(theta) * state[0] + p1[0]
    y = np.sin(theta) * state[0] + p1[1]
    return np.vstack((x, y, state[1:]))

if __name__ == "__main__":
    pass