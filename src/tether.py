import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np
import time

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

        t0 = time.time()
        for _ in self.t:
            for e in self.elements:
                e.step(h)
        print("\033[32mTotal process time : {} s\033[0m".format(time.time()-t0))

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
        ax_ek.fill_between(self.t, np.maximum(np.zeros(m.shape), m - 3*std), m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_ek.plot(self.t, m, color="crimson", linewidth=3, label="mean of potential energy")
        ax_ek.plot(self.t, np.maximum(np.zeros(m.shape), m - 3*std), color="teal", linewidth=2)
        ax_ek.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_ek.set_title("Kinetic Energy")
        ax_ek.grid()
        ax_ek.set_xlabel(r"Time (in $s$)")
        ax_ek.set_ylabel(r"Energy")
        ax_ek.set_xlim(self.t0, self.tf)
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
        _, ax_energy = plt.subplots()
        total_ek = []
        total_ep = []

        for e in self.elements:
            if e.previous is not None and e.next is not None:
                total_ek.append(e.Ek[:-1])
                total_ep.append(e.Ep[:-1])

        total_ep = np.asarray(total_ep)
        total_ek = np.asarray(total_ek)

        energy = total_ek + total_ep
        
        ax_energy.plot(self.t, energy.T, color="grey")

        m, std = np.mean(energy, axis=0), np.std(energy, axis=0)
        ax_energy.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_energy.plot(self.t, m, color="crimson", linewidth=3, label="mean of potential energy")
        ax_energy.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_energy.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_energy.set_title("Energy")
        ax_energy.grid()
        ax_energy.set_xlabel(r"Time (in $s$)")
        ax_energy.set_ylabel(r"Energy")
        ax_energy.set_xlim(self.t0, self.tf)
        ax_energy.legend()
    
    def simulate(self, save=False, filename="tether.mp4"):
        # Attaching 3D axis to the figure 
        self.fig = plt.figure() 
        self.ax = p3.Axes3D(self.fig)

        # Setting up title
        self.fig.suptitle('Tether', fontsize=16)

        # Setting up axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Setting up axis limits and view
        self.ax.set_xlim3d(-1, 4)
        self.ax.set_ylim3d(-6, 6)
        self.ax.set_zlim3d(-20, 0)
        self.ax.view_init(elev=20, azim=-50)

        # Creating n line object for each TetherElement
        self.graph, = self.ax.plot([], [], [], color="teal", marker="o")

        # Creating 3D animation
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=int((self.tf-self.t0)/self.h), interval=self.h*1000, blit=True, repeat=False)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=int(1/self.h), metadata=dict(artist='Me'), bitrate=1800, codec="libx264")
            self.ani.save(filename, writer=writer)

    def animate(self, i):
        X, Y, Z = [], [], []
        for e in self.elements:
            X.append(e.position[i][0][0])
            Y.append(e.position[i][1][0])
            Z.append(e.position[i][2][0])
        self.graph.set_data(np.asarray(X), np.asarray(Y))
        self.graph.set_3d_properties(np.asarray(Z))
        return self.graph,        


if __name__ == "__main__":
    T = Tether(25, 15)
    T.process(0, 45, 1/20)

    # T.monitor_potential_energy()
    # T.monitor_kinetic_energy()
    # T.monitor_energy()
    # T.monitor_length()
    # T.monitor_length_error()
    # T.monitor_angle()
    # plt.show()
    
    T.simulate()
    plt.show()
    T.write_animation()