import numpy as np
import uuid
import yaml

### TODO
# Cleaning the force mask
# Cleaning config.yaml

class TetherElement:
    # Physical constants of the system
    g = 9.81
    rho = 1000

    def __init__(self, mass, length, volume, state, TetherElement_config_filename, is_extremity=False):
        # UUID and pointer to the neighbors TetherElements
        self.uuid = uuid.uuid4()
        self.previous = None
        self.next = None
        self.is_extremity = is_extremity

        # Physical parameters of the TetherElement
        self.mass = mass
        self.length = length
        self.volume = volume

        # State vector of the TetherElement [x, y, z, theta, vx, vy, vz, vtheta].T
        X = np.zeros((8, 1), dtype=np.float64)
        X[:4] = state

        # State vector history
        self.state_history = [X]

        # Energy list
        self.Ek = [0.]
        self.Ep = [0.]

        # Mask to use forces
        self.forces_mask = np.array([True, True, True, True, True, True, True, True])
        self.acceleration_limit = 1e4

        # Length variables
        self.previous_length, self.next_length = self.length, self.length
        self.E_previous, self.E_next = 0., 0.

        # Twist variables
        self.previous_twist, self.next_twist = 0., 0.

        # Bend variables
        self.previous_bend, self.next_bend = self.length, self.length
        self.E_previous, self.E_next = 0., 0.

        # Support vector for torque
        self.v = np.array([[0], [0], [1]])

        # Loading_parameters
        self.parse(TetherElement_config_filename)

    def __str__(self):
        res = "TetherElement : {} \n".format(self.uuid)
        res += "\t Positon : {}\n".format(self.get_position().flatten())

        previous_uuid = ("None" if self.previous is None else str(self.previous.uuid))
        res += "\t Prev \t : {} \n".format(previous_uuid)

        next_uuid = ("None" if self.next is None else str(self.next.uuid))
        res += "\t Next \t : {} \n".format(next_uuid)
        return res
    
    def parse(self, config_filename):
        with open(config_filename) as f:
            parameters = yaml.load(f)

            # Length coefficients
            self.length_Kp = parameters["Length"]["Kp"]
            self.length_Kd = parameters["Length"]["Kd"]
            self.length_Ki = parameters["Length"]["Ki"]

            # Bend coefficients
            self.twist_Kp = parameters["Twist"]["Kp"]
            self.twist_Kd = parameters["Twist"]["Kd"]
            self.twist_Ki = parameters["Twist"]["Ki"]

            # Bend coefficients
            self.bend_Kp = parameters["Bend"]["Kp"]
            self.bend_Kd = parameters["Bend"]["Kd"]
            self.bend_Ki = parameters["Bend"]["Ki"]

            # Drag coefficient
            self.drag_f = parameters["Drag"]["f"]

    def get_position(self, i=None):
        if i is not None:
            return np.asarray(self.state_history[i])[:3]
        else:
            return np.asarray(self.state_history[-1])[:3]

    def get_angle(self, i=None):
        if i is not None:
            return np.asarray(self.state_history[i])[3]
        else:
            return np.asarray(self.state_history[-1])[3]

    def get_velocity(self):
        return np.asarray(self.state_history[-1])[4:7]

    def get_positions(self):
        return np.asarray(self.state_history)[:, :3]
    
    def get_velocities(self):
        return np.asarray(self.state_history)[:, 4:7]

    def get_angles(self):
        return np.asarray(self.state_history)[:, 3]

    def step(self, h):
        # Compute acceleration
        forces = np.hstack((self.Fg(), self.Fb(), self.Ft_prev(h),  self.Ft_next(h), self.Ff(), self.Fs(h), self.Fr_prev(h), self.Fr_next(h)))
        forces = np.sum(forces[:, self.forces_mask], axis=1).reshape(4, 1)
        acceleration = np.clip(forces / self.mass, -self.acceleration_limit, self.acceleration_limit)

        if self.is_extremity:
            U = -acceleration
        else:
            U = np.zeros((4, 1))

        self.state_history.append(self.state_history[-1] + h * np.vstack((self.state_history[-1][4:8], acceleration + U)))
            
        self.Ek.append(self.mass/2*(self.get_velocity().T@self.get_velocity())[0,0])
        self.Ep.append(self.Ep[-1] + self.dW(h))

    def Fg(self):
        return np.array([[0], [0], [-self.mass * self.g], [0]])

    def Fb(self) :
        return np.array([[0], [0], [self.rho * self.volume * self.g], [0]])

    def Ft_prev(self, h):
        if self.previous is not None:
            # Current lenght
            lm = np.linalg.norm(self.get_position() - self.previous.get_position())

            # Force support vector
            u = (self.previous.get_position() - self.get_position()) / lm

            # Error computing
            e = (self.length - lm)
            de = (lm - self.previous_length) / h

            # Processing force
            force = - ( self.length_Kp * e + self.length_Kd * de + self.length_Ki * self.E_previous) * u

            # Updating values
            self.previous_length = lm
            self.E_previous += h * e

            return np.vstack((force, np.zeros((1, 1))))
        else :
            return np.zeros((4, 1))

    def Ft_next(self, h):
        if self.next is not None:
            # Current length
            lm = np.linalg.norm(self.next.get_position() - self.get_position())

            # Force support vector
            u = (self.next.get_position() - self.get_position()) / lm

            # Error computing
            e = (self.length - lm)
            de = (lm - self.next_length) / h

            # Processing force
            force = - ( self.length_Kp * e + self.length_Kd * de + self.length_Ki * self.E_next) * u

            # Updating values
            self.next_length = lm
            self.E_next += h * e
            return np.vstack((force, np.zeros((1, 1))))
        else :
            return np.zeros((4, 1))

    def Ff(self):
        force = - self.drag_f * self.get_velocity()*np.abs(self.get_velocity())
        return np.vstack((force, np.zeros((1, 1))))

    def Fr_prev(self, h):
        if self.previous is None:
            return np.zeros((4, 1))
        e = 2 * np.arctan(np.tan((self.get_angle() - self.previous.get_angle())/2))
        de = (e - self.previous_twist) / h
        torque = - (self.twist_Kp * e + self.twist_Kd * de)
        self.previous_twist = e
        return np.vstack((np.zeros((3, 1)), torque))
    
    def Fr_next(self, h):
        if self.next is None:
            return np.zeros((4, 1))
        e = 2 * np.arctan(np.tan((self.next.get_angle() - self.get_angle())/2))
        de = (e - self.next_twist) / h
        torque = self.twist_Kp * e + self.twist_Kd * de
        self.next_twist = e
        return np.vstack((np.zeros((3, 1)), torque))

    def Fs(self, h):
        return np.zeros((4, 1))
        # if self.next is None or self.previous is None:
        #     return np.zeros((3, 1))
        
        # u_previous = (self.previous.get_position() - self.get_position())
        # u_next = (self.next.get_position() - self.get_position())

        # value = np.clip((u_next.T @ u_previous) / (np.linalg.norm(self.previous.get_position() - self.get_position()) * np.linalg.norm((self.next.get_position() - self.get_position()))), -1.0, 1.0)
        # e = (np.arccos(value) - np.pi/2) / (np.pi / 2)
        # value = np.clip((u_next + h*(self.next.get_velocity() - self.get_velocity())).T @ (u_previous+h*(self.previous.get_velocity() - self.get_velocity())), -1.0, 1.0)
        # de = (np.arccos(value) - e) / h
        # self.E_torque += h*e

        # if np.allclose((u_previous + u_next), np.zeros((3, 1))):
        #     return np.zeros((3, 1))
        # else:
        #     self.v = (u_previous + u_next) / np.linalg.norm(u_previous + u_next)
        #     return - (self.Tp*e + self.Td*de + self.Ti*self.E_torque) * self.v

    def dW(self, h):
        W = self.get_velocity().T @ np.hstack((self.Fg(), self.Fb(), self.Ft_prev(h), self.Ft_next(h), self.Ff(), self.Fs(h)))[:3]
        return np.sum(h * W)


if __name__ == "__main__":
    pass
