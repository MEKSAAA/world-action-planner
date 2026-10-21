"""Validate CPU FK against Genesis 1.3.1 on measured source states.

Run in the RoboWits simulator environment with its checkout on PYTHONPATH.
Only a robot-only scene is constructed; no physics steps or source writes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import Episodes
from .geometry import Marvin


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
    try:
        for index in np.linspace(0, len(state)-1, 7).astype(int):
            s = state[index]
            q = np.r_[s[:7], s[7]/2, s[7]/2, s[8:15], s[15]/2, s[15]/2]
            robot.robot_entity.set_dofs_position(torch.tensor(q, dtype=torch.float32),
                                                dofs_idx_local=robot._seq_pos_to_dof.tolist())
            links = model.forward(s)
            for name, pose in links.items():
                actual = robot.robot_entity.get_link(name).get_pos(relative=False).numpy().reshape(-1, 3)[0]
                errors.append(float(np.linalg.norm(actual-pose[:3, 3])))
        report = {'genesis_version': version('genesis-world'), 'episode': args.episode,
                  'samples': 7, 'link_comparisons': len(errors), 'physics_steps': 0,
                  'max_link_error_m': max(errors), 'passed': max(errors) < 1e-5}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(report)
        if not report['passed']:
            raise RuntimeError('CPU FK does not agree with Genesis')
    finally:
        scene.destroy()


if __name__ == '__main__':
    main()
