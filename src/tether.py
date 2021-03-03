import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
import time
import yaml

from tether_element import TetherElement
from initialization import get_catenary_coefficients, get_initial_position


### TODO
# Fixing n / L / number of TetherElement per meters
# Adding extermities forces monitoring to see forces which are going to be applied to the MMO
# Enhance visual representation
# Using double linked list and exec {for i in range(10): exec("obj{} = Stock(name, price)".format(i))} to instantiate TetherElements

class Tether:
    def __init__(self, Tether_config_filename, TetherElement_config_filename):
        # Parsing configuration file
        self.parse(Tether_config_filename)

        # Double linked list of TetherElement
        self.head = TetherElement(self.element_mass, self.element_length, self.element_volume, self.position_head, TetherElement_config_filename, is_extremity=True)
        self.tail = TetherElement(self.element_mass, self.element_length, self.element_volume, self.position_tail, TetherElement_config_filename, is_extremity=True)

        # Processing initialization parameters
        initial_parameters = get_catenary_coefficients(self.position_head, self.position_tail, self.length)        

        # Initialise double linked list of TetherElement
        self.previous_element = self.head
        for i in range(1, self.n-1):
            position = get_initial_position(self.position_head, self.position_tail, self.length, self.n, i, initial_parameters)
            exec("self.TetherElement{} = TetherElement(self.element_mass, self.element_length, self.element_volume, position, TetherElement_config_filename)".format(i))
            exec("self.TetherElement{}.previous = self.previous_element".format(i))
            exec("self.previous_element.next = self.TetherElement{}".format(i))
            exec("self.previous_element = self.TetherElement{}".format(i))
        exec("self.TetherElement{}.next = self.tail".format(i))
        exec("self.tail.previous = self.TetherElement{}".format(i))

    def __str__(self):
        res = ""
        e = self.head
        while e.next is not None:
            res += str(e)
            e = e.next
        res += str(e)
        return res

    def parse(self, config_filename):
        with open(config_filename) as f:
            parameters = yaml.load(f)

            # Tether parameters parsing
            self.length = parameters["Tether"]["length"]
            self.n = parameters["Tether"]["n"]
            self.linear_mass = parameters["Tether"]["linear_mass"]
            self.linear_volume = parameters["Tether"]["linear_volume"]

            # Environment parameters parsing
            self.rho = parameters["Environment"]["rho"]
            self.g = parameters["Environment"]["g"]

            # First element parsing
            x_first = parameters["Elements"]["position_head"]["x"]
            y_first = parameters["Elements"]["position_head"]["y"]
            z_first = parameters["Elements"]["position_head"]["z"]
            self.position_head = np.array([[x_first], [y_first], [z_first]])

            # Last element parsing
            x_last = parameters["Elements"]["position_tail"]["x"]
            y_last = parameters["Elements"]["position_tail"]["y"]
            z_last = parameters["Elements"]["position_tail"]["z"]
            self.position_tail = np.array([[x_last], [y_last], [z_last]])

            # Other parameters processing
            self.element_length = self.length / (self.n - 1)
            self.element_mass = self.length * self.linear_mass / self.n
            self.element_volume = self.length * self.linear_volume / self.n
        
    def process(self, t0, tf, h):
        # Saving parameters
        self.t0, self.tf, self.h = t0, tf, h
        self.t = np.arange(self.t0, self.tf, self.h)

        t0 = time.time()
        for _ in self.t:
            e = self.head
            while e.next is not None:
                e.step(h)
                e = e.next
            e.step(h)
        print("\033[32mTotal process time : {} s\033[0m".format(time.time()-t0))

    def monitor_length(self):
        fig_length, ax_length = plt.subplots()
        total_length = []

        e = self.head
        while e.next is not None:
            total_length.append(np.linalg.norm(e.next.get_positions() - e.get_positions(), axis=1))
            e = e.next
        
        total_length = np.squeeze(np.asarray(total_length))[:, :-1]
        ax_length.plot(self.t, total_length.T, color="grey")

        m, std = np.mean(total_length, axis=0), np.std(total_length, axis=0)
        ax_length.fill_between(self.t, np.maximum(np.zeros(m.shape), m - 3*std), m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_length.plot(self.t, m, color="crimson", linewidth=3, label="mean of lengths")
        ax_length.plot(self.t, np.maximum(np.zeros(m.shape), m - 3*std), color="teal", linewidth=2)
        ax_length.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_length.plot(self.t, self.element_length*np.ones(self.t.shape), color="orange", label="target length", linewidth=2)

        #ax_length.set_title("Length of the links")
        ax_length.grid()
        ax_length.set_xlabel(r"Time (in $s$)")
        ax_length.set_ylabel(r"Length (in $m$)")
        ax_length.set_xlim(self.t0, self.tf)
        ax_length.legend()

        fig_length.set_size_inches(w=3.5, h=2.8)
        plt.tight_layout()
        
        return fig_length, ax_length

    def monitor_length_error(self):
        fig_length_error, ax_length_error = plt.subplots()
        total_length = []

        e = self.head
        while e.next is not None:
            total_length.append(np.linalg.norm(e.next.get_positions() - e.get_positions(), axis=1))
            e = e.next
        
        total_length = np.squeeze(np.asarray(total_length))[:, :-1]

        m = np.mean(total_length, axis=0)
        relative_error = 100*(m - self.element_length*np.ones(self.t.shape))/self.element_length*np.ones(self.t.shape)
        ax_length_error.fill_between(self.t, np.zeros(self.t.shape), relative_error, color="crimson", alpha=0.4)
        ax_length_error.plot(self.t, relative_error, color="crimson")
    
        # ax_length_error.set_title("Relative error between the target length and the mean length of links")
        ax_length_error.grid()
        ax_length_error.set_xlabel(r"Time (in $s$)")
        ax_length_error.set_ylabel("Relative error (in %)")
        ax_length_error.set_xlim(self.t0, self.tf)

        fig_length_error.set_size_inches(w=3.5, h=2.8)
        plt.tight_layout()

        return fig_length_error, ax_length_error

    def monitor_angle(self):
        fig_angle, ax_angle = plt.subplots()
        total_angle = []

        e = self.head
        while e.next is not None:
            total_angle.append(e.get_angles())
            e = e.next
        
        total_angle = np.squeeze(np.asarray(total_angle))[:, :-1]

        ax_angle.plot(self.t, total_angle.T, color="grey")

        m, std = np.mean(total_angle, axis=0), np.std(total_angle, axis=0)
        ax_angle.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_angle.plot(self.t, m, color="crimson", linewidth=3, label="mean of lengths")
        ax_angle.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_angle.plot(self.t, m + 3*std, color="teal", linewidth=2)

        # ax_angle.set_title("Angle between links")
        ax_angle.grid()
        ax_angle.set_xlabel(r"Time (in $s$)")
        ax_angle.set_ylabel(r"Angle (in $rad$)")
        ax_angle.set_xlim(self.t0, self.tf)
        ax_angle.legend()

        return fig_angle, ax_angle

    def monitor_shape(self):
        fig_shape, ax_shape = plt.subplots()
        total_angle = []

        e = self.head.next
        while e.next is not None:
            u_previous = np.squeeze(np.asarray(e.previous.get_positions())[:-1] - np.asarray(e.get_positions())[:-1])
            u_next = np.squeeze(np.asarray(e.next.get_positions())[:-1] - np.asarray(e.get_positions())[:-1])
            total_angle.append(np.arccos(np.sum((u_previous*u_next) / (np.linalg.norm(u_previous) * np.linalg.norm(u_next)), axis=1)))
            e = e.next
        
        total_angle = np.squeeze(np.asarray(total_angle))

        ax_shape.plot(self.t, total_angle.T, color="grey")

        m, std = np.mean(total_angle, axis=0), np.std(total_angle, axis=0)
        ax_shape.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_shape.plot(self.t, m, color="crimson", linewidth=3, label="mean of lengths")
        ax_shape.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_shape.plot(self.t, m + 3*std, color="teal", linewidth=2)

        ax_shape.plot(self.t, np.pi/2*np.ones(self.t.shape), color="orange", label="reference angle", linewidth=2)

        # ax_angle.set_title("Angle between links")
        ax_shape.grid()
        ax_shape.set_xlabel(r"Time (in $s$)")
        ax_shape.set_ylabel(r"Angle (in $rad$)")
        ax_shape.set_xlim(self.t0, self.tf)
        ax_shape.legend()

        return fig_shape, ax_shape

    def monitor_kinetic_energy(self):
        fig_ek, ax_ek = plt.subplots()
        total_ek = []

        e = self.head.next
        while e.next is not None:
            total_ek.append(e.Ek[:-1])
            e = e.next

        total_ek = np.asarray(total_ek)
        ax_ek.plot(self.t, total_ek.T, color="grey")

        m, std = np.mean(total_ek, axis=0), np.std(total_ek, axis=0)
        ax_ek.fill_between(self.t, np.maximum(np.zeros(m.shape), m - 3*std), m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_ek.plot(self.t, m, color="crimson", linewidth=3, label="mean of kinetic energy")
        ax_ek.plot(self.t, np.maximum(np.zeros(m.shape), m - 3*std), color="teal", linewidth=2)
        ax_ek.plot(self.t, m + 3*std, color="teal", linewidth=2)

        # ax_ek.set_title("Kinetic Energy")
        ax_ek.grid()
        ax_ek.set_xlabel(r"Time (in $s$)")
        ax_ek.set_ylabel(r"Energy (in $J$)")
        ax_ek.set_xlim(self.t0, self.tf)
        ax_ek.legend()

        fig_ek.set_size_inches(w=3.5, h=2.8)
        plt.tight_layout()

        return fig_ek, ax_ek

    def monitor_potential_energy(self):
        fig_ep, ax_ep = plt.subplots()
        total_ep = []

        e = self.head.next
        while e.next is not None:
            total_ep.append(e.Ep[:-1])
            e = e.next

        total_ep = np.asarray(total_ep)
        ax_ep.plot(self.t, total_ep.T, color="grey")

        m, std = np.mean(total_ep, axis=0), np.std(total_ep, axis=0)
        ax_ep.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_ep.plot(self.t, m, color="crimson", linewidth=3, label="mean of potential energy")
        ax_ep.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_ep.plot(self.t, m + 3*std, color="teal", linewidth=2)

        # ax_ep.set_title("Potential Energy")
        ax_ep.grid()
        ax_ep.set_xlabel(r"Time (in $s$)")
        ax_ep.set_ylabel(r"Energy (in $J$)")
        ax_ep.set_xlim(self.t0, self.tf)
        ax_ep.legend()

        fig_ep.set_size_inches(w=3.5, h=2.8)
        plt.tight_layout()
        
        return fig_ep, ax_ep

    def monitor_energy(self):
        fig_energy, ax_energy = plt.subplots()
        total_ek = []
        total_ep = []

        e = self.head.next
        while e.next is not None:
            total_ek.append(e.Ek[:-1])
            total_ep.append(e.Ep[:-1])
            e = e.next

        total_ep = np.asarray(total_ep)
        total_ek = np.asarray(total_ek)

        energy = total_ek + total_ep
        
        ax_energy.plot(self.t, energy.T, color="grey")

        m, std = np.mean(energy, axis=0), np.std(energy, axis=0)
        ax_energy.fill_between(self.t, m - 3*std, m + 3*std, facecolor='teal', alpha=0.4, label=r"$3.\sigma$ area")
        ax_energy.plot(self.t, m, color="crimson", linewidth=3, label="mean of energy")
        ax_energy.plot(self.t, m - 3*std, color="teal", linewidth=2)
        ax_energy.plot(self.t, m + 3*std, color="teal", linewidth=2)

        #ax_energy.set_title("Energy")
        ax_energy.grid()
        ax_energy.set_xlabel(r"Time (in $s$)")
        ax_energy.set_ylabel(r"Energy (in $J$)")
        ax_energy.set_xlim(self.t0, self.tf)
        ax_energy.legend()

        fig_energy.set_size_inches(w=3.5, h=2.8)
        plt.tight_layout()
        
        return fig_energy, ax_energy

    def simulate(self, save=False, filename="tether.mp4"):
        # Attaching 3D axis to the figure 
        self.fig = plt.figure() 
        self.ax = p3.Axes3D(self.fig)
        self.ax.set_proj_type('ortho')

        # Setting up title
        self.fig.suptitle('Tether', fontsize=16)

        # Setting up axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Setting up axis limits and view
        self.ax.set_xlim3d(10, 20)
        self.ax.set_ylim3d(-6, 6)
        self.ax.set_zlim3d(-15, 5)
        self.ax.view_init(elev=20, azim=-50)

        # Creating n line object for each TetherElement
        self.graph, = self.ax.plot([], [], [], color="teal", marker="o", markersize=10)

        # Creating 3D animation
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=int((self.tf-self.t0)/self.h), interval=self.h*1000, blit=True, repeat=False)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=int(1/self.h), metadata=dict(artist='Me'), bitrate=1800, codec="libx264")
            self.ani.save(filename, writer=writer)

    def animate(self, i):
        X, Y, Z, theta = [], [], [], []
        e = self.head
        while e.next is not None:
            X.append(e.get_position(i)[0][0])
            Y.append(e.get_position(i)[1][0])
            Z.append(e.get_position(i)[2][0])
            theta.append(e.get_angle(i)[0])
            e = e.next
        X.append(e.get_position(i)[0][0])
        Y.append(e.get_position(i)[1][0])
        Z.append(e.get_position(i)[2][0])
        theta.append(e.get_angle(i)[0])
        self.graph.set_data(np.asarray(X), np.asarray(Y))
        self.graph.set_3d_properties(np.asarray(Z))
        return self.graph,


if __name__ == "__main__":
    T = Tether("./config/Tether.yaml", "./config/TetherElement.yaml")
    T.process(0, 30, 1/20)

    fig_length_error, ax_length_error = T.monitor_length_error()
    fig_length, ax_length = T.monitor_length()
    plt.show()

    T.simulate()
    plt.show()