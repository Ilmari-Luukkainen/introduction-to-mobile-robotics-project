import numpy as np
from forward_kinematics import fk_pose
from inverse_kinematics import Q_MIN, Q_MAX, inverse_kinematics_position
from so101_mujoco_utils import get_mujoco_ee_pose


def test_ik():
    # 5 reachable test targets joint angles in degrees
    reachable_targets = [
        ("Target 1", [30, -20, 45, 10, 60]),
        ("Target 2", [-45, -35, 50, -15, 20]),
        ("Target 3", [55, 25, -35, 30, -40]),
        ("Target 4", [15, -55, 80, -25, 0]),
        ("Target 5", [-75, 15, 30, -20, 45]),
    ]

    q_init = np.zeros(5)

    print("REACHABLE TARGETS ")
    for name, angles_deg in reachable_targets:
        # Convert angles to radians and get target position and rotation from forward kinematics
        angles_rad = np.deg2rad(angles_deg)
        target_pos, target_rotation = fk_pose(angles_rad)

        # Solve inverse kinematics
        q_sol, converged, iters = inverse_kinematics_position(target_pos, q_init, return_info=True)
        
        # Verify result in the MuJoCo simulation environment
        mujoco_pos, mujoco_rotation = get_mujoco_ee_pose(q_sol)
        error_mm = np.linalg.norm(target_pos - mujoco_pos) * 1000

        print(name, "| Target:", np.round(target_pos, 3), "| Iter:", iters, "| MuJoCo Error (mm):", round(error_mm, 4))

        assert converged is True
        assert error_mm < 0.1
        assert np.all(q_sol >= Q_MIN) and np.all(q_sol <= Q_MAX)

    # 1target outside the robot's workspace reach
    print("\n UNREACHABLE TARGET")
    unreachable_pos = np.array([0.80, 0.0, 0.35])
    q_unreach, converged_unreach, iters_unreach = inverse_kinematics_position(unreachable_pos, q_init, return_info=True)
    
    mujoco_unreach, mujoco_unreach_rot = get_mujoco_ee_pose(q_unreach)
    unreach_error_mm = np.linalg.norm(unreachable_pos - mujoco_unreach) * 1000

    print("Unreachable | Target:", unreachable_pos, "| Converged:", converged_unreach, "| Iter:", iters_unreach, "| Error (mm):", round(unreach_error_mm, 2))

    assert converged_unreach is False
    assert np.all(q_unreach >= Q_MIN) and np.all(q_unreach <= Q_MAX)

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_ik()