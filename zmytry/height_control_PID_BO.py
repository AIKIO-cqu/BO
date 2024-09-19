import matplotlib.pyplot as plt
from skopt import gp_minimize


class Drone:

    def __init__(self):
        self.height = 0.0
        self.velocity = 0.0
        self.mass = 1.0  # Mass of the drone in kg
        self.gravity = 9.81  # Gravity acceleration in m/s^2

    def get_height(self):
        return self.height

    def set_thrust(self, thrust, dt):
        # Simulate applying the control signal to the drone
        net_force = thrust - self.mass * self.gravity
        acceleration = net_force / self.mass
        self.velocity += acceleration * dt
        self.height += self.velocity * dt


class PIDController:

    def __init__(self, kp, ki, kd):
        # Initialize PID controller gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Initialize error and integral term
        self.error = 0.0
        self.integral = 0.0
        self.previous_error = 0.0

    def calculate(self, target_height, current_height, dt):
        # Update error and integral term
        self.error = target_height - current_height
        self.integral += self.error * dt
        derivative = (self.error - self.previous_error) / dt

        # Calculate control signal using PID formula
        control_signal = (
            self.kp * self.error + self.ki * self.integral + self.kd * derivative
        )

        # Update previous error
        self.previous_error = self.error

        return control_signal

    # Add a method to reset the kp, ki, and kd values
    def reset(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd


def objective_P(params):
    kp = params[0]
    drone = Drone()
    pid_controller = PIDController(kp=kp, ki=2.0, kd=2.0)

    height_values = []

    target_height = 10.0

    dt = 0.1  # Time step

    for _ in range(200):
        current_height = drone.get_height()
        thrust = pid_controller.calculate(target_height, current_height, dt)
        drone.set_thrust(thrust, dt)
        height_values.append(current_height)

    return sum([(h - target_height) ** 2 for h in height_values])


def get_optimal_P():
    result = gp_minimize(objective_P, [(0.1, 10.0)], n_calls=20)
    return result.x[0]


def objective_I(params):
    ki = params[0]
    drone = Drone()
    pid_controller = PIDController(kp=0.85, ki=ki, kd=2.0)

    height_values = []

    target_height = 10.0

    dt = 0.1  # Time step

    for _ in range(200):
        current_height = drone.get_height()
        thrust = pid_controller.calculate(target_height, current_height, dt)
        drone.set_thrust(thrust, dt)
        height_values.append(current_height)

    return sum([(h - target_height) ** 2 for h in height_values])


def get_optimal_I():
    result = gp_minimize(objective_I, [(0.1, 10.0)], n_calls=40)
    return result.x[0]


def objective_D(params):
    kd = params[0]
    drone = Drone()
    pid_controller = PIDController(kp=0.85, ki=2.0, kd=kd)

    height_values = []

    target_height = 10.0

    dt = 0.1  # Time step

    for _ in range(200):
        current_height = drone.get_height()
        thrust = pid_controller.calculate(target_height, current_height, dt)
        drone.set_thrust(thrust, dt)
        height_values.append(current_height)

    return sum([(h - target_height) ** 2 for h in height_values])


def get_optimal_D():
    result = gp_minimize(objective_D, [(0.1, 10.0)], n_calls=40)
    return result.x[0]


def objective(params):
    kp, ki, kd = params[0], params[1], params[2]
    drone = Drone()
    pid_controller = PIDController(kp=kp, ki=ki, kd=kd)

    height_values = []

    target_height = 10.0

    dt = 0.1  # Time step

    for _ in range(200):
        current_height = drone.get_height()
        thrust = pid_controller.calculate(target_height, current_height, dt)
        drone.set_thrust(thrust, dt)
        height_values.append(current_height)

    return sum([(h - target_height) ** 2 for h in height_values])


def get_optimal():
    result = gp_minimize(objective, [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0)], n_calls=40)
    return result.x


# Modify the main function to include plotting
def main():
    # Initialize drone and PID controller
    drone = Drone()

    # Get optimal PID parameters
    optimal_params = get_optimal()
    print("Optimal PID parameters:", optimal_params)
    pid_controller = PIDController(optimal_params[0], optimal_params[1], optimal_params[2])

    height_values = []

    target_height = 10.0

    dt = 0.1  # Time step

    for _ in range(200):
        # Get current height from drone
        current_height = drone.get_height()

        # Calculate control signal using PID controller
        thrust = pid_controller.calculate(target_height, current_height, dt)

        # Apply control signal to drone
        drone.set_thrust(thrust, dt)

        # Append height and time values to the lists
        height_values.append(current_height)

    # Plot the drone's height
    plt.plot(height_values, label="Drone Height")
    plt.axhline(y=target_height, color="r", linestyle="--", label="Target Height")
    plt.xlabel("Time Steps")
    plt.ylabel("Height")
    plt.title("Drone Height")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
