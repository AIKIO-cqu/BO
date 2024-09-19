import matplotlib.pyplot as plt


class Drone:

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.mass = 1.0  # Mass of the drone in kg
        self.friction = 0.1  # Friction coefficient

    def get_position(self):
        return self.x, self.y

    def set_thrust(self, thrust_x, thrust_y, dt):
        # Simulate applying the control signal to the drone
        fx = thrust_x - self.friction * self.vx
        ax = fx / self.mass
        self.vx += ax * dt
        self.x += self.vx * dt

        fy = thrust_y - self.friction * self.vy
        ay = fy / self.mass
        self.vy += ay * dt
        self.y += self.vy * dt


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

    def calculate(self, target_position, current_position, dt):
        # Update error and integral term
        self.error = target_position - current_position
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


# Modify the main function to include plotting
def main():
    # Initialize drone and PID controller
    drone = Drone()

    pid_controller_x = PIDController(kp=4.5, ki=2.0, kd=2.0)
    pid_controller_y = PIDController(kp=4.5, ki=2.0, kd=2.0)

    x_position_values = []
    y_position_values = []

    x_target_position = 10.0
    y_target_position = 15.0

    dt = 0.1  # Time step

    for _ in range(200):
        # Get current position from drone
        current_position_x, current_position_y = drone.get_position()

        # Calculate control signal using PID controller
        thrust_x = pid_controller_x.calculate(x_target_position, current_position_x, dt)
        thrust_y = pid_controller_y.calculate(y_target_position, current_position_y, dt)

        # Apply control signal to drone
        drone.set_thrust(thrust_x, thrust_y, dt)

        # Append position and time values to the lists
        x_position_values.append(current_position_x)
        y_position_values.append(current_position_y)

    # Plot the drone's position and target position
    plt.figure()
    plt.plot(x_position_values, label="X Position")
    plt.plot(y_position_values, label="Y Position")
    plt.axhline(
        y=x_target_position, color="r", linestyle="--", label="X Target Position"
    )
    plt.axhline(
        y=y_target_position, color="g", linestyle="--", label="Y Target Position"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
