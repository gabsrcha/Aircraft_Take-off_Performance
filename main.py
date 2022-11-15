"""
- 2D Simulation of an aircraft's takeoff performance from ground roll, pitch up and 1st climb up to 35ft
"""

import os.path
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

save_path = r"D:\Documents-HDD\Aircraft_Take-off_Performance\Data"
save_path = os.path.join(save_path, str(time.strftime("%d-%m-%y %H-%M-%S")))
os.mkdir(save_path)

takeoff_data = pd.DataFrame(columns=["t", "X", "Z", "Pitch", "alpha", "gamma", "V_x", "V_z", "a_x", "a_z", "L", "P",
                                     "W", "D", "F_f", "R_NW", "R_LG", "M_pitch", "M_W", "M_L", "M_P", "M_T", "M_NW",
                                     "T"])

# Aerodynamic conditions
rho = 1.225  # Kg/m^3

# Aircraft Config
mass = 70e3  # Kg
W = mass * -9.81  # N

# Fuselage
R_fuselage = 4
L_fuselage = 40

# Take-off Thrust
TWR = 0.3
T_TO = TWR * mass * 9.81  # N

# Wing
S_wing = 152  # m^2
C_Lo = 0.8
C_Lalpha = 0.7
AR_wing = 15
epsilon_wing = 0.8

# Horizontal stabiliser
S_Hstab = 20  # m^2
C_Po = -0.3
C_Palpha = -0.03
AR_Hstab = 8
epsilon_Hstab = 0.8

# Elevator
deflection_TO = 3
C_P_def = -1

# Drag
C_Do_wing = 0.02
C_Do_Hstab = 0.02
# C_Do_Fuselage = 0.02

# Moment arms [m]
x_NW = 3
x_CG = 17
x_AC_L = 18
x_LG = 20
x_AC_P = 37
z_T = 0.5

# Tire friction
mu = 0

# Simulation
dt = 0.01  # s

# Take-off speeds
V_one = 40  # m/s
V_r = 60  # m/s


def main() -> None:
    # Simulation variables
    t = 0
    X, Z = 0, 0
    V_x, V_z = 0, 0
    theta, alpha, gamma = 0, 0, 0
    q = 0
    elevator_def = 0

    I_yy = 1/4*mass*R_fuselage**2 + 1/12*mass*L_fuselage**2  # Cylinder approximation

    while Z < 10.668:  # 35ft

        if t > 120:
            break

        if V_x > V_r:
            elevator_def = np.radians(deflection_TO)

        # Dynamic pressure
        P_dyn = 0.5 * rho * abs(V_x) ** 2

        # Wing Lift
        C_L = C_Lo + C_Lalpha * alpha
        L = P_dyn * S_wing * C_L

        # H_stab lift
        C_P = C_Po + C_Palpha * alpha + C_P_def * elevator_def
        P = P_dyn * S_Hstab * C_P

        # Drag
        C_Di_wing = C_L ** 2 / (np.pi * epsilon_wing * AR_wing)
        C_Di_Hstab = C_P ** 2 / (np.pi * epsilon_Hstab * AR_Hstab)
        D = P_dyn * S_wing * (C_Do_wing + C_Di_wing) + P_dyn * S_Hstab * (C_Do_Hstab + C_Di_Hstab)

        # Reaction forces
        R_NW = (W*(x_LG - x_CG) + P*(x_LG - x_AC_P) + L*(x_LG - x_AC_L) + T_TO*z_T) / (x_NW - x_LG)
        if R_NW < 0:
            R_NW = 0

        R_LG = -P*np.cos(theta) - L*np.cos(theta) - R_NW - W
        if R_LG < 0:
            R_LG = 0

        # Friction
        F_f = mu * (R_NW + R_LG)

        # Compute Linear Accelerations
        F_x = (T_TO*np.cos(theta) - D*np.cos(gamma) - F_f)
        F_z = L*np.cos(theta) + P*np.cos(theta) + R_NW + R_LG + W
        a_x = F_x / mass
        a_z = F_z / mass
        # if a_z < 0:
        #     a_z = 0

        # Linear Integration
        V_x = V_x + a_x * dt
        V_z = V_z + a_z * dt

        X = X + V_x * dt
        Z = Z + V_z * dt

        # Compute moments
        if round(Z, 3) > 0:
            M_L = L*(x_CG-x_AC_L)
            M_T = T_TO*z_T
            M_P = P*(x_CG-x_AC_P)
            M_theta = M_L + M_T + M_P   # M_CG
        else:
            M_L = L*(x_LG - x_AC_L)
            M_T = T_TO*z_T
            M_P = P*(x_LG - x_AC_P)
            M_W = W*np.cos(theta)*(x_LG - x_CG)
            M_NW = R_NW * (x_LG - x_NW)
            M_theta = M_L + M_T + M_P + M_W + M_NW

        # Pitch
        q = q + (M_theta / I_yy) * dt
        theta = round(theta + q * dt, 4)
        if theta > 0:
            theta = theta % (2*np.pi)
        else:
            theta = theta % (-2*np.pi)

        if Z > 0:
            gamma = np.arctan(V_z / V_x)
            alpha = theta - gamma
        else:
            gamma = 0
            alpha = theta

        # Store data
        takeoff_data.loc[len(takeoff_data)] = [t, X, Z, np.degrees(theta), np.degrees(alpha), np.degrees(gamma), V_x,
                                               V_z, a_x, a_z, L, P, W, D, F_f, R_NW, R_LG, M_theta, M_W, M_L, M_P, M_T,
                                               M_NW, T_TO]

        # Increment time
        t += dt

        print(np.degrees(theta))

    print(takeoff_data)

    plt.rcParams['axes.grid'] = True

    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    axes[0].plot(takeoff_data.t, takeoff_data.X)
    plt.setp(axes[0], ylabel="X [m]")
    plt.setp(axes[0], xlabel="Time [s]")
    axes[1].plot(takeoff_data.t, takeoff_data.Z)
    plt.setp(axes[1], ylabel="Z [m]")
    plt.setp(axes[1], xlabel="Time [s]")
    plt.savefig(os.path.join(save_path, "Displacement"))
    plt.close()

    plt.figure()
    plt.plot(takeoff_data.t, takeoff_data.V_x, label="V_x")
    plt.plot(takeoff_data.t, takeoff_data.V_z, label="V_z")
    plt.title("Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Velocity"))
    plt.close()

    plt.figure()
    plt.plot(takeoff_data.t, takeoff_data.a_x, label="a_x")
    plt.plot(takeoff_data.t, takeoff_data.a_z, label="a_z")
    plt.title("Acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Acceleration"))
    plt.close()

    plt.figure()
    plt.plot(takeoff_data.t, takeoff_data.Pitch, label="pitch")
    plt.plot(takeoff_data.t, takeoff_data.alpha, label="alpha")
    plt.plot(takeoff_data.t, takeoff_data.gamma, label="gamma")
    plt.title("Attitude")
    plt.xlabel("Time [s]")
    plt.ylabel("Attitude [Deg]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Attitude"))
    plt.close()

    plt.figure()
    plt.plot(takeoff_data.t, takeoff_data.L, label="Wing Lift")
    plt.plot(takeoff_data.t, takeoff_data.P, label="Hstab Lift")
    plt.plot(takeoff_data.t, takeoff_data.R_NW, label="R_Nose Wheel")
    plt.plot(takeoff_data.t, takeoff_data.R_LG, label="R_Landing Gear")
    plt.title("Vertical Forces")
    plt.xlabel("Time [s]")
    plt.ylabel("Vertical Forces [N]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Vertical Forces"))
    plt.close()

    plt.figure()
    plt.plot(takeoff_data.t, takeoff_data["T"], label="Thrust")
    plt.plot(takeoff_data.t, takeoff_data.D, label="Drag")
    plt.plot(takeoff_data.t, takeoff_data.F_f, label="Tire Friction")
    plt.title("Horizontal Forces")
    plt.xlabel("Time [s]")
    plt.ylabel("Horizontal Forces [N]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Horizontal Forces"))
    plt.close()

    plt.figure()
    plt.plot(takeoff_data.t, takeoff_data.M_L, label="M_Wing Lift")
    plt.plot(takeoff_data.t, takeoff_data.M_P, label="M_Hstab Lift")
    plt.plot(takeoff_data.t, takeoff_data.M_W, label="M_Weight")
    plt.plot(takeoff_data.t, takeoff_data.M_T, label="M_Thrust")
    plt.plot(takeoff_data.t, takeoff_data.M_NW, label="M_Nose Wheel")
    plt.title("Moments")
    plt.xlabel("Time [s]")
    plt.ylabel("Moments [Nm]")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Moments"))
    plt.close()

    takeoff_data.to_csv(os.path.join(save_path, "Data.csv"))


if __name__ == "__main__":
    main()
