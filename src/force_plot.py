import numpy as np
import matplotlib.pyplot as plt


def unitary_arrow(a = 0.8):
    M = np.array([[0, a, a, 1, a, a], [0, 0, (1-a)/2, 0, (a-1)/2, 0], [0, 0, 0, 0, 0, 0]])
    return M

def get_force_arrow(p, a):
    M = unitary_arrow()
    return M

if __name__ == "__main__":
    plt.figure()
    M = unitary_arrow(a=0.8)
    plt.plot(M[:, 0], M[:, 1])
    plt.axis("equal")
    plt.show()