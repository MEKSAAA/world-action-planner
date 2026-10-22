"""Validate CPU FK against Genesis 1.3.1 on measured source states.

Run in the RoboWits simulator environment with its checkout on PYTHONPATH.
Only a robot-only scene is constructed; no physics steps or source writes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import Episodes
from .geometry import Marvin, BASE
from .prepare_policy_data import ee14
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    from importlib.metadata import version
    if version('genesis-world') != '1.3.1':
        raise RuntimeError('Validation requires Genesis 1.3.1')
    import genesis as gs
    import torch
    from gs_gym.robots.bimanual_marvin import BimanualMarvinRobot
    from gs_gym.robots.schema import ManipulatorRobotArgs
    source = Episodes(args.source)
    record = next(r for r in source.records() if r['episode_index'] == args.episode)
    state, _ = source.states(record)
    gs.init(backend=gs.cpu, logging_level='warning')
    scene = gs.Scene(show_viewer=False)
    robot = BimanualMarvinRobot(1, scene, ManipulatorRobotArgs(type='bimanual_marvin', position=(0, 0, 1.08)),
                               torch.device('cpu'), urdf_path=str(args.urdf))
    scene.build(n_envs=1)
    robot.init_dof_indices()
    model = Marvin(args.urdf)
    errors = []
    ee_position_errors, ee_rotation_errors = [], []
    ik_position_errors, ik_rotation_errors = [], []
    try:
        for index in np.linspace(0, len(state)-1, 7).astype(int):
            s = state[index]
            q = np.r_[s[:7], s[7]/2, s[7]/2, s[8:15], s[15]/2, s[15]/2]
            robot.robot_entity.set_dofs_position(torch.tensor(q, dtype=torch.float32),
                                                dofs_idx_local=robot._seq_pos_to_dof.tolist())
            links = model.forward(s)
            converted = ee14(model, s[None])[0]
            for side, pos_slice, rot_slice in (('right', slice(0, 3), slice(6, 9)),
                                               ('left', slice(3, 6), slice(9, 12))):
                actual = getattr(robot, f'{side}_ee_pose').numpy().reshape(7)
                ee_position_errors.append(float(np.linalg.norm(actual[:3]-BASE-converted[pos_slice])))
                actual_rot = Rotation.from_quat(actual[[4, 5, 6, 3]])
                predicted_rot = Rotation.from_rotvec(converted[rot_slice])
                ee_rotation_errors.append(float((actual_rot.inv()*predicted_rot).magnitude()))
            action = torch.tensor(converted, dtype=torch.float32)[None]
            right, left = robot.run_bimanual_ik(action[:, :3], robot._axis_angle_to_quat(action[:, 6:9]),
                                                action[:, 3:6], robot._axis_angle_to_quat(action[:, 9:12]))
            solved = np.r_[right.numpy().reshape(7), s[7], left.numpy().reshape(7), s[15]]
            actual_ee = ee14(model, solved[None])[0]
            ik_position_errors.append(float(np.max(np.abs(actual_ee[:6]-converted[:6]))))
            for part in (slice(6, 9), slice(9, 12)):
                ik_rotation_errors.append(float((Rotation.from_rotvec(actual_ee[part]).inv()*Rotation.from_rotvec(converted[part])).magnitude()))
            for name, pose in links.items():
                actual = robot.robot_entity.get_link(name).get_pos(relative=False).numpy().reshape(-1, 3)[0]
                errors.append(float(np.linalg.norm(actual-pose[:3, 3])))
        report = {'genesis_version': version('genesis-world'), 'episode': args.episode,
                  'samples': 7, 'link_comparisons': len(errors), 'physics_steps': 0,
                  'max_link_error_m': max(errors),
                  'max_ee_position_error_m': max(ee_position_errors),
                  'ee_position_frame': 'robot-base-relative (native observation and Pink IK target)',
                  'max_ee_rotation_error_rad': max(ee_rotation_errors),
                  'max_native_ik_position_error_m': max(ik_position_errors),
                  'max_native_ik_rotation_error_rad': max(ik_rotation_errors),
                  'passed': max(errors+ee_position_errors+ee_rotation_errors) < 1e-5
                            and max(ik_position_errors+ik_rotation_errors) < 1e-3}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(report)
        if not report['passed']:
            raise RuntimeError('CPU FK does not agree with Genesis')
    finally:
        scene.destroy()


if __name__ == '__main__':
    main()
