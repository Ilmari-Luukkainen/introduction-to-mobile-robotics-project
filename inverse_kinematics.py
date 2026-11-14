import numpy as np
import modern_robotics as mr
from forward_kinematics import forward_kinematics, fk_pose
from jacobian import space_jacobian, position_jacobian


def dls(J, lam, error):
    # Compute joint angle update step using Damped Least Squares (DLS)
    I = np.eye(J.shape[1])
    delta_q = np.linalg.inv(J.T @ J + (lam ** 2) * I) @ J.T @ error
    return delta_q


def inverse_kinematics_position(target_pos, q_init, lam=0.1, steps=50):
    # Numerical IK for 3D target position [x, y, z]
    q = np.array(q_init, dtype=float)

    for _ in range(steps):
        current_pos, _ = fk_pose(q)
        error = target_pos - current_pos
        J = position_jacobian(q)
        q = q + dls(J, lam, error)

    return q


if __name__ == "__main__":
    # Test with the example configuration from forward_kinematics.py
    q_true = np.deg2rad([30, -20, 45, 10, 60])
    target_pos, _ = fk_pose(q_true)

    q_init = np.zeros(5)
    q_sol = inverse_kinematics_position(target_pos, q_init)
    reached_pos, _ = fk_pose(q_sol)

    print("Target position  =", np.round(target_pos, 4))
    print("Reached position =", np.round(reached_pos, 4))
    print("Solved angles (rad) =", np.round(q_sol, 4))