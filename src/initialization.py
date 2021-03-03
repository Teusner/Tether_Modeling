import numpy as np
from scipy.optimize import fsolve

def catenary(p, p1, pn, length):
    eq1 = p[0]*np.sinh((pn[0, 0]+p[1])/p[0]) - p[0]*np.sinh((p1[0, 0]+p[1])/p[0]) - length
    eq2 = p[0]*np.cosh((p1[0, 0]+p[1])/p[0]) + p[2] - p1[1, 0]
    eq3 = p[0]*np.cosh((pn[0, 0]+p[1])/p[0]) + p[2] - pn[1, 0]
    return [eq1, eq2, eq3]

def get_catenary_coefficients(p1, pn, length):
    return fsolve(catenary, (1., -(p1[0, 0]+pn[0, 0])/2, (p1[2, 0]+pn[2, 0])/2), args=(p1, pn, length))

def initial_position(p, p1, pn, L, n, i, initial_parameters):
    eq1 = initial_parameters[0]*np.sinh((p[0]+initial_parameters[1])/initial_parameters[0]) - initial_parameters[0]*np.sinh((p1[0, 0]+initial_parameters[1])/initial_parameters[0]) - i * L / (n )
    eq2 = p1[1, 0] + i * (pn[1, 0] - p1[1, 0]) / (n - 1) - p[1]
    eq3 = initial_parameters[0]*np.cosh((p[0]+initial_parameters[1])/initial_parameters[0]) + initial_parameters[2] - p[2]
    eq4 = p1[3, 0] + i * (pn[3, 0] - p1[3, 0]) / (n - 1) - p[3]
    return [eq1, eq2, eq3, eq4]

def get_initial_position(p1, pn, L, n, i, initial_parameters):
    position = fsolve(initial_position, (p1 + pn) / 2, args=(p1, pn, L, n, i, initial_parameters))
    return position.reshape(4, 1)

if __name__ == "__main__":
    pass