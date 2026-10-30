import numpy as np
import mujoco


_MODEL_XML = """
<mujoco model="so101_fk">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="joint1" pos="0.039 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 -1" damping="1"/>
      <body name="joint2" pos="0.030 0 0.116">
        <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" damping="1"/>
        <body name="joint3" pos="0.028 0 0.113">
          <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
          <joint name="elbow_flex" type="hinge" axis="0 1 0" damping="1"/>
          <body name="joint4" pos="0.135 0 0.005">
            <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
            <joint name="wrist_flex" type="hinge" axis="0 1 0" damping="1"/>
            <body name="joint5" pos="0.061 0 0">
              <inertial pos="0 0 0" mass="1" diaginertia="0.001 0.001 0.001"/>
              <joint name="wrist_roll" type="hinge" axis="-1 0 0" damping="1"/>
              <site name="end_effector" pos="0.098 0 -0.008" size="0.005"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_MODEL = mujoco.MjModel.from_xml_string(_MODEL_XML)
_DATA = mujoco.MjData(_MODEL)
_SITE_ID = mujoco.mj_name2id(
    _MODEL, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
)


def get_mujoco_ee_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the MuJoCo end-effector position and rotation for joint angles."""
    angles = np.asarray(q, dtype=float)
    if angles.shape != (5,):
        raise ValueError(f"Expected five joint angles, got shape {angles.shape}")

    _DATA.qpos[:5] = angles
    mujoco.mj_forward(_MODEL, _DATA)
    position = _DATA.site_xpos[_SITE_ID].copy()
    rotation = _DATA.site_xmat[_SITE_ID].reshape(3, 3).copy()
    return position, rotation