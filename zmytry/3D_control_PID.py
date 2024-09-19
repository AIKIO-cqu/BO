import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation


class Drone:

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.mass = 1.0  # Mass of the drone in kg
        self.gravity = 9.81  # Gravity acceleration in m/s^2
        self.friction = 0.1  # Friction coefficient

    def get_position(self):
        return self.x, self.y, self.z

    def set_thrust(self, thrust_x, thrust_y, thrust_z, dt):
        # x control
        fx = thrust_x - self.friction * self.vx
        ax = fx / self.mass
        self.vx += ax * dt
        self.x += self.vx * dt

        # y control
        fy = thrust_y - self.friction * self.vy
        ay = fy / self.mass
        self.vy += ay * dt
        self.y += self.vy * dt

        # z control
        fz = thrust_z - self.mass * self.gravity
        az = fz / self.mass
        self.vz += az * dt
        self.z += self.vz * dt


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


# Modify the main function to include plotting
def main():
    # Initialize drone and PID controller
    drone = Drone()

    # Initialize PID controllers for x, y, and z axes
    pid_controller_x = PIDController(kp=4.5, ki=2.0, kd=2.0)
    pid_controller_y = PIDController(kp=4.5, ki=2.0, kd=2.0)
    pid_controller_z = PIDController(kp=4.5, ki=2.0, kd=2.0)

    x_position_values = []
    y_position_values = []
    z_position_values = []

    x_target_position = 7.0
    y_target_position = 12.0
    z_target_position = 6.0

    dt = 0.01  # Time step

    T = 10.0  # Total time
    t_all = np.arange(0, T, dt)  # 用于绘制动画的时间序列

    for i in range(int(T / dt)):
        # Update time
        t_all[i] = i * dt

        # Get current height from drone
        current_position_x, current_position_y, current_position_z = (
            drone.get_position()
        )

        # Calculate control signal using PID controller
        thrust_x = pid_controller_x.calculate(x_target_position, current_position_x, dt)
        thrust_y = pid_controller_y.calculate(y_target_position, current_position_y, dt)
        thrust_z = pid_controller_z.calculate(z_target_position, current_position_z, dt)

        # Apply control signal to drone
        drone.set_thrust(thrust_x, thrust_y, thrust_z, dt)

        # Append height and time values to the lists
        x_position_values.append(current_position_x)
        y_position_values.append(current_position_y)
        z_position_values.append(current_position_z)

    # Plot the drone's position and target position
    plt.figure()
    plt.title("Drone Position Control")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.plot(t_all, x_position_values, label="X Position")
    plt.plot(t_all, y_position_values, label="Y Position")
    plt.plot(t_all, z_position_values, label="Z Position")
    plt.axhline(
        y=x_target_position, color="r", linestyle="--", label="X Target Position"
    )
    plt.axhline(
        y=y_target_position, color="g", linestyle="--", label="Y Target Position"
    )
    plt.axhline(
        y=z_target_position, color="b", linestyle="--", label="Z Target Position"
    )
    plt.legend()
    plt.show()

    # 动画绘制
    show_animation(
        t_all,
        dt,
        x_position_values,
        y_position_values,
        z_position_values,
        x_target_position,
        y_target_position,
        z_target_position,
    )


def show_animation(t_all, dt, x_list, y_list, z_list, x_traget, y_traget, z_traget):
    numFrames = 4  # 每隔4帧绘制一次
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)

    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    (line,) = ax.plot([], [], [], "--", lw=1, color="blue")  # 四旋翼的飞行轨迹

    mid_x = (x_list.max() + x_list.min()) * 0.5
    mid_y = (y_list.max() + y_list.min()) * 0.5
    mid_z = (z_list.max() + z_list.min()) * 0.5
    maxRange = (
        np.array(
            [
                x_list.max() - x_list.min(),
                y_list.max() - y_list.min(),
                z_list.max() - z_list.min(),
            ]
        ).max()
        * 0.5
        + 0.5
    )
    ax.set_xlim3d([mid_x - maxRange, mid_x + maxRange])
    ax.set_xlabel("X")
    ax.set_ylim3d([mid_y - maxRange, mid_y + maxRange])
    ax.set_ylabel("Y")
    ax.set_zlim3d([mid_z - maxRange, mid_z + maxRange])
    ax.set_zlabel("Z")

    # 显示起始点和目标点
    ax.scatter(x_list[0], y_list[0], z_list[0], c="g", marker="o", label="Start Point")
    ax.scatter(x_traget, y_traget, z_traget, c="r", marker="o", label="Target Point")
    ax.legend(loc="upper right")

    # 显示时间
    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def updateLines(i):
        # 核心函数，定义了动画每一帧的更新逻辑
        time = t_all[i * numFrames]

        # 提取从模拟开始到当前帧的所有x、y、z坐标，用于绘制四旋翼的飞行轨迹
        x_from0 = x_list[0 : i * numFrames]
        y_from0 = y_list[0 : i * numFrames]
        z_from0 = z_list[0 : i * numFrames]

        line.set_data(x_from0, y_from0)
        line.set_3d_properties(z_from0)

        titleTime.set_text("Time = {:.2f} s".format(time))

    ani = animation.FuncAnimation(
        fig=fig,
        func=updateLines,
        frames=len(t_all[0:-1:numFrames]),  # 动画总帧数
        interval=dt * numFrames * 1000,  # 每帧间隔时间(ms)
        blit=False,
    )
    plt.show()
    return ani


if __name__ == "__main__":
    main()
