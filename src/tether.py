import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from tether_element import TetherElement

### TODO
# Adding process time benchmark
# Adding extermities forces to see forces which are going to be applied to the MMO
# Better behavioral force and torque with correct PID
# Monitoring angles with normal vectors : angle = arccos(a.b)


class Tether:
    def __init__(self, L, n):
        # Tether parameters
        self.n = n
        self.L = L

        # List of TetherElements
        self.elements = []

        # TetherElements parameters
        self.element_length = self.L / (self.n - 1)
        self.element_mass = 5 * self.element_length
        self.element_volume = np.pi*0.005**2*self.element_length

        # Extremities
        self.position_first = np.array([[12.], [0.], [5.]])
        self.position_last = np.array([[15.], [0.], [0.]])

        # Initialise random positions for each TetherElements
        for i in range(n):
            xe = i * (self.position_last[0] - self.position_first[0]) / n
            ye = i * (self.position_last[1] - self.position_first[1]) / n
            ze = i * (self.position_last[2] - self.position_first[2]) / n
            position = np.array([xe, ye, ze]) #+ np.random.randn(3, 1)
            self.elements.append(TetherElement(self.element_mass, self.element_length, self.element_volume, position))
        
        # Chaining elements
        for i in range(1, n-1):
            self.elements[i].previous = self.elements[i-1]
            self.elements[i].next = self.elements[i+1]

        self.elements[0].next = self.elements[1]
        self.elements[-1].previous = self.elements[-2]

    def __str__(self):
        res = ""
        for e in self.elements:
            res += str(e)
        return res        

    def process(self, t0, tf, h):
        # Saving parameters
        self.t0, self.tf, self.h = t0, tf, h
        self.t = np.arange(self.t0, self.tf, self.h)

        for _ in self.t:
            for e in self.elements:
                e.step(h)

    def simulate(self):
        # Attaching 3D axis to the figure 
        self.fig = plt.figure() 
        self.ax = p3.Axes3D(self.fig) 

        # Setting the axes properties 
        self.ax.set_xlim3d(0, 12)
        self.ax.set_ylim3d(-6, 6)
        self.ax.set_zlim3d(0, 12)

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=int((self.tf-self.t0)/self.h), interval=int(1/self.h), blit=False, repeat=False)
    
    def write_animation(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1/self.h), metadata=dict(artist='Me'), bitrate=1800, codec="libx264")
        self.ani.save('tether_1.mp4', writer=writer)

    def monitor_length(self):
        _, ax_length = plt.subplots()
        total_length = []

        for e in self.elements:
            if e.next is not None:
                total_length.append(np.linalg.norm(np.asarray(e.next.position)[:-1] - np.asarray(e.position)[:-1], axis=1))
        
        total_length = np.squeeze(np.asarray(total_length))
        ax_length.plot(self.t, total_length.T, color="grey")

        m, std = np.mean(total_length, axis=0), np.std(total_length, axis=0)
        ax_length.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_length.plot(self.t, m, color="crimson", linewidth=3, label="mean of lengths")
        ax_length.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_length.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_length.plot(self.t, self.element_length*np.ones(self.t.shape), color="orange", label="target length", linewidth=2)

        ax_length.set_title("Length of the links")
        ax_length.grid()
        ax_length.set_xlabel(r"Time (in $s$)")
        ax_length.set_ylabel(r"Length (in $m$)")
        ax_length.set_xlim(self.t0, self.tf)
        ax_length.legend()

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 0, 600, 450)

    def monitor_length_error(self):
        _, ax_length_error = plt.subplots()
        total_length = []

        for e in self.elements:
            if e.next is not None:
                total_length.append(np.linalg.norm(np.asarray(e.next.position)[:-1] - np.asarray(e.position)[:-1], axis=1))
        
        total_length = np.squeeze(np.asarray(total_length))

        m = np.mean(total_length, axis=0)
        relative_error = 100*(m - self.element_length*np.ones(self.t.shape))/self.element_length*np.ones(self.t.shape)
        ax_length_error.fill_between(self.t, np.zeros(self.t.shape), relative_error, color="crimson", alpha=0.4)
        ax_length_error.plot(self.t, relative_error, color="crimson")
    
        ax_length_error.set_title("Relative error between the target length and the mean length of links")
        ax_length_error.grid()
        ax_length_error.set_xlabel(r"Time (in $s$)")
        ax_length_error.set_ylabel("Relative error (in %)")
        ax_length_error.set_xlim(self.t0, self.tf)

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(600, 0, 800, 450)

    def monitor_angle(self):
        _, ax_angle = plt.subplots()
        total_angle = []

        for e in self.elements:
            if e.previous is not None and e.next is not None:
                u_previous = np.squeeze(np.asarray(e.previous.position)[:-1] - np.asarray(e.position)[:-1])
                u_next = np.squeeze(np.asarray(e.next.position)[:-1] - np.asarray(e.position)[:-1])
                total_angle.append(np.arccos(np.sum((u_previous*u_next) / (np.linalg.norm(u_previous) * np.linalg.norm(u_next)), axis=1)))
        
        total_angle = np.squeeze(np.asarray(total_angle))

        ax_angle.plot(self.t, total_angle.T, color="grey")

        m, std = np.mean(total_angle, axis=0), np.std(total_angle, axis=0)
        ax_angle.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_angle.plot(self.t, m, color="crimson", linewidth=3, label="mean of lengths")
        ax_angle.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_angle.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_angle.plot(self.t, np.pi/2*np.ones(self.t.shape), color="orange", label="reference angle", linewidth=2)

        ax_angle.set_title("Angle between links")
        ax_angle.grid()
        ax_angle.set_xlabel(r"Time (in $s$)")
        ax_angle.set_ylabel(r"Angle (in $rad$)")
        ax_angle.set_xlim(self.t0, self.tf)
        ax_angle.legend()

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 0, 600, 450)

    def monitor_kinetic_energy(self):
        _, ax_ek = plt.subplots()
        total_ek = []

        for e in self.elements:
            if e.previous is not None and e.next is not None:
                total_ek.append(e.Ek[:-1])

        total_ek = np.asarray(total_ek)
        ax_ek.plot(self.t, total_ek.T, color="grey")

        m, std = np.mean(total_ek, axis=0), np.std(total_ek, axis=0)
        ax_ek.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_ek.plot(self.t, m, color="crimson", linewidth=3, label="mean of potential energy")
        ax_ek.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_ek.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_ek.set_title("Kinetic Energy")
        ax_ek.grid()
        ax_ek.set_xlabel(r"Time (in $s$)")
        ax_ek.set_ylabel(r"Energy")
        ax_ek.set_xlim(self.t0, self.tf)
        ax_ek.set_ylim(-0.2, 0.2)
        ax_ek.legend()

    def monitor_potential_energy(self):
        _, ax_ep = plt.subplots()
        total_ep = []

        for e in self.elements:
            if e.previous is not None and e.next is not None:
                total_ep.append(e.Ep[:-1])

        total_ep = np.asarray(total_ep)
        ax_ep.plot(self.t, total_ep.T, color="grey")

        m, std = np.mean(total_ep, axis=0), np.std(total_ep, axis=0)
        ax_ep.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_ep.plot(self.t, m, color="crimson", linewidth=3, label="mean of potential energy")
        ax_ep.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_ep.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_ep.set_title("Potential Energy")
        ax_ep.grid()
        ax_ep.set_xlabel(r"Time (in $s$)")
        ax_ep.set_ylabel(r"Energy")
        ax_ep.set_xlim(self.t0, self.tf)
        ax_ep.legend()

    def monitor_energy(self):
        # Creating a plot
        self.fig_energy, self.ax_energy = plt.subplots()

        self.total_ek = []
        self.total_ep = []

        # Showing energy of each nodes
        for e in self.elements:
            if e.previous is not None and e.next is not None:
                self.total_ek.append(e.Ek[1:])
                self.total_ep.append(e.Ep[1:])

        self.total_ek = np.asarray(self.total_ek)
        self.total_ep = np.asarray(self.total_ep)

        self.energy = self.total_ek + self.total_ep
        
        self.ax_energy.plot(self.t, self.energy.T, color="grey")

        m, std = np.mean(self.energy, axis=0), np.std(self.energy, axis=0)
        self.ax_energy.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        self.ax_energy.plot(self.t, m, color="crimson", linewidth=3, label="mean of potential energy")
        self.ax_energy.plot(self.t, m - 3*std, color="teal", linewidth=2)
        self.ax_energy.plot(self.t, m + 3*std, color="teal", linewidth=2)

        self.ax_energy.set_title("Energy")
        self.ax_energy.grid()
        self.ax_energy.set_xlabel(r"Time (in $s$)")
        self.ax_energy.set_ylabel(r"Energy")
        self.ax_energy.set_xlim(self.t0, self.tf)
        self.ax_energy.legend()
    
    def animate(self, i): 
        self.ax.clear() 
        self.ax.set_xlim3d(0, 20)
        self.ax.set_ylim3d(-10, 10)
        self.ax.set_zlim3d(-15, 5)
                    
        self.ax.plot3D(self.S[i, 0], self.S[i, 1], self.S[i, 2], color="teal")

        for k in range(self.n):
            if k == 0:
                col = "purple"
            elif k == self.n-1:
                col = "gold"
            else:
                col = "crimson"
            self.ax.scatter3D(self.S[i, 0, k], self.S[i, 1, k], self.S[i, 2, k], color=col)


if __name__ == "__main__":
    T = Tether(25, 15)
    T.process(0, 45, 1/20)
    # T.monitor_potential_energy()
    # T.monitor_kinetic_energy()
    # T.monitor_energy()
    # T.monitor_length()
    # T.monitor_length_error()
    T.monitor_angle()
    plt.show()
    
    # T.simulate()
    # plt.show()
    # T.write_animation()