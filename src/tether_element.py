import numpy as np
import uuid

class TetherElement:
    # Physical constants of the system
    g = 9.81
    rho = 1000

    def __init__(self, mass, length, volume, position):
        # UUID and pointer to the neighbors TetherElements
        self.uuid = uuid.uuid4()
        self.previous = None
        self.next = None

        # Physical parameters of the TetherElement
        self.mass = mass
        self.length = length
        self.volume = volume

        # Mechanical parameters of the TetherElement
        self.position = position
        self.velocity = np.zeros((3, 1), dtype=np.float64)
        self.acceleration = np.array((3, 1), dtype=np.float64)

        # Energy
        self.Ek = [0.]
        self.Ep = [0.]

        # Mask to use forces
        self.forces_mask = np.array([True, True, True, True, True, True])
        self.acceleration_limit = 1e4

        # Coefficient for the behavioral model
        self.kp = 200
        self.kd = 3.5
        self.ki = 3
        self.previous_length = 0.
        self.next_length = 0.
        self.previous_int = 0.
        self.next_int = 0.

        # Proportionnal resistant torque
        self.Tp = 50.

    def __str__(self):
        res = "TetherElement : {} \n".format(self.uuid)
        res += "\t Positon : {}\n".format(self.position.flatten())

        previous_uuid = ("None" if self.previous is None else str(self.previous.uuid))
        res += "\t Prev \t : {} \n".format(previous_uuid)

        next_uuid = ("None" if self.next is None else str(self.next.uuid))
        res += "\t Next \t : {} \n".format(next_uuid)
        return res

    def step(self, h):
        if self.previous is not None and self.next is not None:
            forces = np.hstack((self.Fg(), self.Fb(), self.Ft_prev(h),  self.Ft_next(h), self.Ff(), self.Fs()))
            self.acceleration = np.clip(1 / self.mass * ((forces[:, self.forces_mask]) @ np.ones((6, 1))), -self.acceleration_limit, self.acceleration_limit)
            self.velocity += h * self.acceleration
            self.position += h * self.velocity

            # Energy processing
            self.Ek.append(self.mass/2*(self.velocity.T@self.velocity)[0,0])
            self.Ep.append(self.Ep[-1] + self.dW(h))

    def Fg(self):
        return np.array([[0], [0], [-self.mass * self.g]])

    def Fb(self) :
        return np.array([[0], [0], [self.rho * self.volume * self.g]])

    def Ft_prev(self, h):
        if self.previous is not None:
            lm = np.linalg.norm(self.position - self.previous.position)
            u = (self.previous.position - self.position) / lm
            force = - ( self.kp * (self.length - lm) / self.length + self.kd * (lm - self.previous_length) / h + self.ki * self.previous_int) * u
            self.previous_length = lm
            self.previous_int += h * (self.length - lm)
            return force
        else :
            return np.zeros((3, 1))

    def Ft_next(self, h):
        if self.next is not None:
            lm = np.linalg.norm(self.next.position - self.position)
            u = (self.next.position - self.position) / lm
            force = - ( self.kp * (self.length - lm) / self.length + self.kd * (lm - self.next_length) / h + self.ki * self.next_int) * u
            self.next_length = lm
            self.next_int += h * (self.length - lm)
            return force
        else :
            return np.zeros((3, 1))

    def Ff(self):
        return - self.velocity*np.abs(self.velocity)

    def Fs(self):
        if self.next is not None and self.previous is not None:
            u_previous = self.previous.position - self.position
            u_next = self.next.position - self.position

            v = (u_previous + u_next) / 2
            return self.Tp * v
        else :
            return np.zeros((3, 1))

    def dW(self, h):
        W = self.velocity.T @ np.hstack((self.Fg(), self.Fb(), self.Ft_prev(h), self.Ft_next(h), self.Ff(), self.Fs()))
        return np.sum(h * W)


if __name__ == "__main__":
    t1 = TetherElement(1, 1, 1, np.array([[0], [0], [1]]))
    t2 = TetherElement(1, 1, 1, np.array([[0], [0], [1]]))
    t1.next = t2
    t2.previous = t1
    print(t1)
    print(t2)