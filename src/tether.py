import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation 


g = 9.81
rho = 1000

class Tether:
    def __init__(self, L, n):
        self.n = n
        self.L = L

        self.elements = []

        self.element_length = self.L / (self.n - 1)
        self.element_mass = 0.1 * self.element_length
        self.element_volume = np.pi*0.005**2*self.element_length

        # Extremities
        self.position_first = np.array([[0.], [0.], [0.]])
        self.position_last = np.array([[10.], [0.], [10.]])

        # Chaining elements
        for i in range(n):
            xe = i * (self.position_last[0] - self.position_first[0]) / n
            ye = i * (self.position_last[1] - self.position_first[1]) / n
            ze = i * (self.position_last[2] - self.position_first[2]) / n
            position = np.array([xe, ye, ze]) + np.random.randn(3, 1)
            self.elements.append(TetherElement(self.element_mass, self.element_length, self.element_volume, position))
        
        for i in range(1, n-1):
            self.elements[i].previous = self.elements[i-1]
            self.elements[i].next = self.elements[i+1]

        self.elements[0].next = self.elements[1]
        self.elements[-1].previous = self.elements[-2]

    def __str__(self):
        res = ""
        for i, e in enumerate(self.elements):
            res += "Element {:d} : \n".format(i)
            res += "\t Prev \t : {} \n".format(e.previous)
            res += "\t Next \t : {} \n".format(e.next)
        return res

    def step(self, h):
        for e in self.elements:
            e.step(h)

    def process(self, t0, tf, h):
        # Saving parameters
        self.t0, self.tf, self.h = t0, tf, h
        self.S = []

        self.t = np.arange(self.t0, self.tf, self.h)

        for i in self.t:
            self.step(self.h)
            x_points, y_points, z_points = [], [], []
            for e in self.elements:
                x_points.append(e.position[0, 0])
                y_points.append(e.position[1, 0])
                z_points.append(e.position[2, 0])
            self.S.append([x_points, y_points, z_points])
        self.S = np.asarray(self.S)

    def simulate(self):
        # Attaching 3D axis to the figure 
        self.fig = plt.figure() 
        self.ax = p3.Axes3D(self.fig) 

        # Setting the axes properties 
        self.ax.set_xlim3d(0, 12)
        self.ax.set_ylim3d(-6, 6)
        self.ax.set_zlim3d(0, 12)

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=int((self.tf-self.t0)/self.h), interval=int(1/self.h), blit=False, repeat=False)
    
    def show(self):
        plt.show()
    
    def write_animation(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1/self.h), metadata=dict(artist='Me'), bitrate=1800)
        self.ani.save('tether.mp4', writer=writer)

    def monitor_length(self, i):
        # Creating a plot
        self.fig_length, self.ax_length = plt.subplots()

        # Showing the length of the ith link
        length = np.linalg.norm(self.S[:, :, i+1]-self.S[:, :, i], axis=1)
        self.ax_length.plot(self.t, length, label="link length")
        self.ax_length.plot(self.t, self.element_length*np.ones(self.t.shape), label="target length")

        self.ax_length.set_title(r"Length of the ${}th$ link".format(i+1))
        self.ax_length.grid()
        self.ax_length.set_xlabel(r"Time (in $s$)")
        self.ax_length.set_ylabel(r"Length (in $m$)")
        self.ax_length.set_xlim(self.t0, self.tf)
        #self.ax_length.set_ylim(0)
        self.ax_length.legend()
        plt.show()
    
    def animate(self, i): 
        self.ax.clear() 
        self.ax.set_xlim3d(0, 12)
        self.ax.set_ylim3d(-6, 6)
        self.ax.set_zlim3d(0, 12)
                    
        self.ax.plot3D(self.S[i, 0], self.S[i, 1], self.S[i, 2], color="teal")

        for k in range(self.n):
            if k == 0:
                col = "purple"
            elif k == self.n-1 is None:
                col = "gold"
            else:
                col = "crimson"
            self.ax.scatter3D(self.S[i, 0, k], self.S[i, 1, k], self.S[i, 2, k], color=col)


class TetherElement:
    def __init__(self, mass, length, volume, position):
        self.previous = None
        self.next = None

        self.position = position
        self.velocity = np.zeros((3, 1), dtype=np.float64)
        self.acceleration = np.array((3, 1), dtype=np.float64)

        self.forces_mask = np.array(3*[[True, True, True, True, True]])

        self.mass = mass
        self.length = length
        self.volume = volume

        self.kp = 10

    def step(self, h):
        if self.previous is not None and self.next is not None:
            forces = np.hstack((self.Fg(), self.Fb(), self.Ft_prev(),  self.Ft_next(), self.F_f()))
            self.acceleration = np.clip(1 / self.mass * ((self.forces_mask * forces) @ np.ones((5, 1))), -1e5, 1e5)
            self.velocity += h * self.acceleration
            self.position += h * self.velocity

    def Fg(self):
        return np.array([[0], [0], [self.mass * g]])

    def Fb(self) :
        '''
        To be replaced with Omega which return the volume of the immerged tether element which is not always the complete volume
        '''
        return np.array([[0], [0], [- rho * self.volume * g]])

    def Ft_prev(self):
        if self.previous is not None:
            lm = np.linalg.norm(self.position - self.previous.position)
            u = (self.previous.position - self.position) / lm
            return - self.kp * (self.length - lm) / self.length * u

    def Ft_next(self):
        if self.next is not None:
            lm = np.linalg.norm(self.next.position - self.position)
            u = (self.next.position - self.position) / lm
        return - self.kp * (self.length - lm) / self.length * u

    def F_f(self):
        return - self.velocity*np.abs(self.velocity)

if __name__ == "__main__":
    T = Tether(25, 10)
    T.process(0, 50, 1/10)
    T.monitor_length(5)
    #T.simulate()
    #T.write_animation()