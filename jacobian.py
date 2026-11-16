import numpy as np
import modern_robotics as mr
from forward_kinematics import Slist, forward_kinematics, fk_pose


def space_jacobian(q):
    # Compute the 6x5 Space Jacobian matrix
    Js = mr.JacobianSpace(Slist, q)
    return Js


def body_jacobian(q):
    # Compute the 6x5 Body Jacobian matrix in the end-effector frame
    Js = mr.JacobianSpace(Slist, q)
    T = forward_kinematics(q)
    Jb = mr.Adjoint(mr.TransInv(T)) @ Js
    return Jb


def position_jacobian(q):
    # Compute the 3x5 Position Jacobian for 3D [x, y, z] velocity
    Js = mr.JacobianSpace(Slist, q)
    p, _ = fk_pose(q)
    J_pos = -mr.VecToso3(p) @ Js[0:3, :] + Js[3:6, :]
    return J_pos


if __name__ == "__main__":
    # Test with the example configuration from forward_kinematics.py
    q = np.deg2rad([30, -20, 45, 10, 60])

    print("Js =")
    print(np.round(space_jacobian(q), 3))
    print("\nJb =")
    print(np.round(body_jacobian(q), 3))
    print("\nJ_pos =")
    print(np.round(position_jacobian(q), 3))