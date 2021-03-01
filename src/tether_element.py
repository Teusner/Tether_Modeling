import numpy as np
import uuid
import yaml

### TODO
# Cleaning the force mask
# Cleaning the acceleration processing part using array
# Better integration of acceleration and velocity
# Adding orientation to state of TetherElement
# Cleaning config.yaml
# Adding Tether twist moment along the Tether

class TetherElement:
    # Physical constants of the system
    g = 9.81
    rho = 1000

    def __init__(self, mass, length, volume, position, config_filename=None):
        # UUID and pointer to the neighbors TetherElements
        self.uuid = uuid.uuid4()
        self.previous = None
        self.next = None

        # Physical parameters of the TetherElement
        self.mass = mass
        self.length = length
        self.volume = volume

        # Mechanical parameters of the TetherElement
        self.position = [position]
        self.velocity = [np.zeros((3, 1), dtype=np.float64)]
        self.acceleration = [np.array((3, 1), dtype=np.float64)]

        # Energy list
        self.Ek = [0.]
        self.Ep = [0.]

        # Mask to use forces
        self.forces_mask = np.array([True, True, True, True, True, True])
        self.acceleration_limit = 1e4

        # Coefficient for the behavioral model
        self.kp = 200
        self.kd = 3.5
        self.ki = 3
        self.previous_length = self.length
        self.next_length = self.length
        self.previous_int = 0.
        self.next_int = 0.

        # Integrator for force
        self.E_previous = 0.
        self.E_next = 0.

        # Integrator for torque
        self.E_torque = 0.

        # Support vector for torque
        self.v = np.array([[0], [0], [1]])

        # Proportionnal resistant torque
        self.Tp = 50.

        # Loading_parameters
        if config_filename is not None:
            self.load_parameters(config_filename)

    def __str__(self):
        res = "TetherElement : {} \n".format(self.uuid)
        res += "\t Positon : {}\n".format(self.get_position().flatten())

        previous_uuid = ("None" if self.previous is None else str(self.previous.uuid))
        res += "\t Prev \t : {} \n".format(previous_uuid)

        next_uuid = ("None" if self.next is None else str(self.next.uuid))
        res += "\t Next \t : {} \n".format(next_uuid)
        return res
    
    def load_parameters(self, filename):
        with open(filename) as f:
            parameters = yaml.load(f)

            # TetherElement parameters
            self.mass = parameters["TetherElement"]["mass"]
            self.length = parameters["TetherElement"]["length"]
            self.volume = parameters["TetherElement"]["volume"]

            # Force
            self.kp = parameters["Length"]["Kp"]
            self.kd = parameters["Length"]["Kd"]
            self.ki = parameters["Length"]["Ki"]

            # Force
            self.Tp = parameters["Torque"]["Kp"]
            self.Td = parameters["Torque"]["Kd"]
            self.Ti = parameters["Torque"]["Ki"]

            # Drag coefficient
            self.f = parameters["Drag"]["f"]

    def get_position(self):
        return self.position[-1]

    def get_velocity(self):
        return self.velocity[-1]

    def get_acceleration(self):
        return self.acceleration[-1]

    def step(self, h):
        # Compute acceleration
        forces = np.hstack((self.Fg(), self.Fb(), self.Ft_prev(h),  self.Ft_next(h), self.Ff(), self.Fs(h)))
        self.acceleration.append(np.clip(1 / self.mass * ((forces[:, self.forces_mask]) @ np.ones((6, 1))), -self.acceleration_limit, self.acceleration_limit))
        
        if self.previous is not None and self.next is not None:
            self.velocity.append(self.get_velocity() + h * self.get_acceleration())
            self.position.append(self.get_position() + h * self.get_velocity())
        else:
            self.velocity.append(self.get_velocity())
            self.position.append(self.get_position())
            
        self.Ek.append(self.mass/2*(self.get_velocity().T@self.get_velocity())[0,0])
        self.Ep.append(self.Ep[-1] + self.dW(h))

    def Fg(self):
        return np.array([[0], [0], [-self.mass * self.g]])

    def Fb(self) :
        return np.array([[0], [0], [self.rho * self.volume * self.g]])

    def Ft_prev(self, h):
        if self.previous is not None:
            lm = np.linalg.norm(self.get_position() - self.previous.get_position())
            u = (self.previous.get_position() - self.get_position()) / lm
            force = - ( self.kp * (self.length - lm) + self.kd * (lm - self.previous_length) / h + self.ki * self.previous_int) * u
            self.previous_length = lm
            self.previous_int += h * (self.length - lm) / lm
            return force
        else :
            return np.zeros((3, 1))

    def Ft_next(self, h):
        if self.next is not None:
            lm = np.linalg.norm(self.next.get_position() - self.get_position())
            u = (self.next.get_position() - self.get_position()) / lm
            force = - ( self.kp * (self.length - lm) + self.kd * (lm - self.next_length) / h + self.ki * self.next_int) * u
            self.next_length = lm
            self.next_int += h * (self.length - lm) / lm
            return force
        else :
            return np.zeros((3, 1))

    def Ff(self):
        return - self.f * self.get_velocity()*np.abs(self.get_velocity())

    def Fs(self, h):
        if self.next is None or self.previous is None:
            return np.zeros((3, 1))
        
        u_previous = (self.previous.get_position() - self.get_position())
        u_next = (self.next.get_position() - self.get_position())

        value = np.clip((u_next.T @ u_previous) / (np.linalg.norm(self.previous.get_position() - self.get_position()) * np.linalg.norm((self.next.get_position() - self.get_position()))), -1.0, 1.0)
        e = (np.arccos(value) - np.pi/2) / (np.pi / 2)
        value = np.clip((u_next + h*(self.next.get_velocity() - self.get_velocity())).T @ (u_previous+h*(self.previous.get_velocity() - self.get_velocity())), -1.0, 1.0)
        de = (np.arccos(value) - e) / h
        self.E_torque += h*e

        if np.allclose((u_previous + u_next), np.zeros((3, 1))):
            return np.zeros((3, 1))
        else:
            self.v = (u_previous + u_next) / np.linalg.norm(u_previous + u_next)
            return - (self.Tp*e + self.Td*de + self.Ti*self.E_torque) * self.v

    def dW(self, h):
        W = self.velocity[-1].T @ np.hstack((self.Fg(), self.Fb(), self.Ft_prev(h), self.Ft_next(h), self.Ff(), self.Fs(h)))
        return np.sum(h * W)


if __name__ == "__main__":
    t1 = TetherElement(1, 1, 1, np.array([[0], [0], [1]]))
    t2 = TetherElement(1, 1, 1, np.array([[0], [0], [1]]))
    t1.next = t2
    t2.previous = t1
    print(t1)
    print(t2)

    t1.load_parameters("./config/TetherElement.yaml")
    print(t2.kp, t2.kd, t2.ki)
