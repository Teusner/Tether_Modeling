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

    T = Tether(25, 10, "./config/TetherElement.yaml")
    T.process(0, 50, 1/20)

    T.monitor_length_error()
    T.monitor_potential_energy()
    T.monitor_kinetic_energy()
    T.monitor_energy()
    T.monitor_length()
    T.monitor_angle()