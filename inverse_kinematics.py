import numpy as np
import modern_robotics as mr
from forward_kinematics import forward_kinematics, fk_pose
from jacobian import space_jacobian, position_jacobian

# Joint limits in radians for the 5 joints of SO-101
Q_MIN = np.deg2rad([-110, -100, -100, -95, -160])
Q_MAX = np.deg2rad([110, 100, 100, 95, 160])


def dls(J, lam, error):
    # Compute joint angle update step using DLS
    I = np.eye(J.shape[1])
    delta_q = np.linalg.inv(J.T @ J + (lam ** 2) * I) @ J.T @ error
    return delta_q


def inverse_kinematics_position(target_pos, q_init, lam=0.01, steps=200, tol=1e-4, max_step=0.25, return_info=False):
    # Numerical inverse kinematics for 3D target position [x, y, z]
    q = np.clip(np.array(q_init, dtype=float), Q_MIN, Q_MAX)
    converged = False

    for step in range(steps):
        # Compute current end effector position and rotation using forward kinematics
        current_pos, rotation = fk_pose(q)
        error = target_pos - current_pos

        # Convergence check stop if position error is below tolerance
        if np.linalg.norm(error) < tol:
            converged = True
            break

        J = position_jacobian(q)

        # Increase damping near singular or poorly conditioned configurations
        s = np.linalg.svd(J, compute_uv=False)
        lam_eff = 0.25 if s[-1] < 0.04 else lam

        delta_q = dls(J, lam_eff, error)

        # Limit maximum step size for solver stability
        step_norm = np.linalg.norm(delta_q)
        if step_norm > max_step:
            delta_q = delta_q * (max_step / step_norm)

        # Update joint angles and enforce joint limits
        q = np.clip(q + delta_q, Q_MIN, Q_MAX)

    if return_info:
        return q, converged, step + 1
    return q


if __name__ == "__main__":
    # Test with the example configuration from forward_kinematics.py
    q_true = np.deg2rad([30, -20, 45, 10, 60])
    target_pos, rotation = fk_pose(q_true)

    q_init = np.zeros(5)
    q_sol = inverse_kinematics_position(target_pos, q_init)
    reached_pos, reached_rotation = fk_pose(q_sol)

    print("Target position  =", np.round(target_pos, 4))
    print("Reached position =", np.round(reached_pos, 4))
    print("Solved angles (rad) =", np.round(q_sol, 4))