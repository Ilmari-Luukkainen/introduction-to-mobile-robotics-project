import numpy as np
import modern_robotics as mr

# Home configuration
M = np.array([[0, 0, 1, 0.391],
              [0, 1, 0, 0],
              [-1, 0, 0, 0.226],
              [0, 0, 0, 1]])

# Screw axes of the joints
Slist = np.array([[0, 0, 0, 0, -1],
                  [0, 1, 1, 1, 0],
                  [-1, 0, 0, 0, 0],
                  [0, -0.116, -0.229, -0.234, 0],
                  [0.039, 0, 0, 0, -0.234],
                  [0, 0.069, 0.097, 0.232, 0]])


def forward_kinematics(q):
    # Compute the end-effector transformation matrix
    T = mr.FKinSpace(M, Slist, q)
    return T


def fk_pose(q):
    # Get position and rotation from the transformation matrix
    T = forward_kinematics(q)
    position = T[0:3, 3]
    rotation = T[0:3, 0:3]
    return position, rotation


if __name__ == "__main__":
    # Test with an example configuration
    q = np.deg2rad([30, -20, 45, 10, 60])
    T = forward_kinematics(q)
    print("T =")
    print(T)
    print("\n")

    position, rotation = fk_pose(q)
    print("position =", position)
    print("\n")
    print("rotation =")
    print(rotation)