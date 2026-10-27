"""Inference adapter for Amber's LIBERO-recipe EEF checkpoint.

All action math is delegated to unmodified hier_policy.eef from the selected
RoboWits worktree. No training code or normalization is changed.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .closed_loop import RobotProxy, same_state, snapshot, state16
from .geometry import pack


def measured_pose(robot):
    joints = state16(robot)
    base = np.asarray(robot.args.position)
    positions, rotations = [], []
    for link in (robot.ee_link_right, robot.ee_link_left):
        positions.append(link.get_pos(relative=False).detach().cpu().numpy().reshape(3)-base)
        q = link.get_quat(relative=False).detach().cpu().numpy().reshape(4)
        rotations.append(Rotation.from_quat(q[[1, 2, 3, 0]]).as_rotvec())
    return np.concatenate(positions + rotations + [np.array([joints[7], joints[15]])]).astype(np.float32)


def chunk_targets(start, deltas):
    from hier_policy.eef import apply_eef_delta
    deltas = np.asarray(deltas)
    if deltas.shape != (10, 14) or not np.isfinite(deltas).all():
        raise ValueError('Expected finite (10,14) EEF policy output')
    target = np.asarray(start, dtype=np.float32).copy()
    targets = []
    for delta in deltas:
        target = apply_eef_delta(target, delta)
        targets.append(target.copy())
    return np.asarray(targets)


def control_action(current, target):
    from hier_policy.eef import eef_between, to_env_ee_delta
    return to_env_ee_delta(eef_between(current, target))


class EEFProxy(RobotProxy):
    def predict_targets(self, env, targets):
        import torch
        before = snapshot(env)
        step = env._step_count
        self.scene.scene.reset()
        source, dest = env._robot.robot_entity, self.robot.robot_entity
        dest.set_qpos(source.get_qpos().clone(), zero_velocity=False)
        dest.set_dofs_velocity(source.get_dofs_velocity().clone())
        self.robot._ik_solver = None
        if not np.allclose(measured_pose(self.robot), measured_pose(env._robot), atol=2e-5, rtol=0):
            raise RuntimeError('Proxy/native measured EE poses differ at initialization')
        states = []
        # WAP requires 20 future frames (one second). Hold the last intended pose
        # after the 10 policy steps; never repeat a motion delta to invent actions.
        for i in range(30):
            action = control_action(measured_pose(self.robot), targets[min(i, len(targets)-1)])
            self.robot.apply_action(torch.as_tensor(action, device=self.robot.device)[None])
            self.scene.scene.step()
            states.append(state16(self.robot))
        if step != env._step_count or not same_state(before, snapshot(env)):
            raise RuntimeError('Candidate imagination mutated the real scene')
        self.last_states = np.asarray(states)
        indices = np.abs(np.arange(1, 31)[:, None]*env.dt-np.arange(1, 21)[None]/20).argmin(axis=0)
        return np.stack([pack(self.fk.skeleton(states[i])) for i in indices])
