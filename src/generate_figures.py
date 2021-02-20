import matplotlib
import matplotlib.pyplot as plt
from tether_element import TetherElement
from tether import Tether

if __name__ == "__main__":
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': '9',
        'ytick.alignment': 'baseline'
    })

    # Simulation
    T = Tether(25, 10, "./config/TetherElement.yaml")
    T.process(0, 50, 1/20)

    # Length figure
    fig_length, ax_length = T.monitor_length()
    plt.savefig('./documentation/plots/length.pgf')

    # Relative error figure
    fig_length_error, ax_length_error = T.monitor_length_error()
    plt.savefig('./documentation/plots/error_length.pgf')

    # Kinetic energy figure
    fig_ek, ax_ek = T.monitor_kinetic_energy()
    plt.savefig('./documentation/plots/kinetic_energy.pgf')

    # Potential energy figure
    fig_ep, ax_ep = T.monitor_potential_energy()
    plt.savefig('./documentation/plots/potential_energy.pgf')    

    # Energy figure
    fig_e, ax_e = T.monitor_energy()
    plt.savefig('./documentation/plots/energy.pgf')

    # Angle figure
    fig_angle, ax_angle = T.monitor_angle()
    plt.savefig('./documentation/plots/angle.pgf')

    # Simulation figure
    T.simulate()
    T.animate(int((T.tf-T.t0)/T.h))
    fig_simulation, ax_simulation = T.fig, T.ax
    T.fig.suptitle('')
    fig_simulation.set_size_inches(w=4.5, h=3.2)
    fig_simulation.savefig('./documentation/plots/simulation.pgf')