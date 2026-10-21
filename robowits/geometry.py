"""Marvin FK and WAP skeletons; no physics or task-object state is used.

Camera/joint conventions: RoboWits Genesis 1.3.1, f5d68f2. Inputs are measured
state[t], never action[t], which commands a subsequent state.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ("ego", "wrist_right", "wrist_left")
PANELS = ("ego", "ego", "wrist_right", "wrist_left")
SIZE = 224
BASE = np.array([0., 0., 1.08])


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def look_at(pos, target, up=(0, 0, 1)):
    pos, target, up = (np.asarray(x, dtype=float) for x in (pos, target, up))
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    t[:3, 3] = pos
    return t


def camera_poses(links):
    offset = np.array([[-1, 0, 0, .009], [0, 1, 0, .04931],
                       [0, 0, -1, .06159 - .0037 + .002], [0, 0, 0, 1.]])
    return {
        "ego": (look_at((.1468, -.0175, 1.4), (.4208, -.0175, 1.0735)), 65.),
        "wrist_right": (links["Pika_Gripper_Base_R"] @ offset, 58.),
        "wrist_left": (links["Pika_Gripper_Base_L"] @ offset, 58.),
    }


class Marvin:
    def __init__(self, urdf):
        self.urdf = Path(urdf)
        root = ET.parse(urdf).getroot()
        self.joints = []
        children = set()
        for j in root.findall("joint"):
            if j.find("mimic") is not None:
                raise ValueError("Expected explicit 18-DOF Marvin URDF without mimic joints")
            origin, axis = j.find("origin"), j.find("axis")
            attr = {} if origin is None else origin.attrib
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            self.joints.append((j.get("name"), j.get("type"), parent, child,
                transform(np.fromstring(attr.get("xyz", "0 0 0"), sep=" "),
                          np.fromstring(attr.get("rpy", "0 0 0"), sep=" ")),
                np.fromstring("1 0 0" if axis is None else axis.get("xyz"), sep=" ")))
            children.add(child)
        roots = {x.get("name") for x in root.findall("link")} - children
        if len(roots) != 1:
            raise ValueError("Expected a single rooted URDF tree")
        self.root = roots.pop()
        required = {f"Joint{i}_{side}" for side in ("R", "L") for i in range(1, 10)}
        if {j[0] for j in self.joints if j[1] != "fixed"} != required:
            raise ValueError("URDF does not match RoboWits 16D state contract")

    def forward(self, state):
        state = np.asarray(state)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 16D right-then-left measured state")
        q = {}
        for offset, side in ((0, "R"), (8, "L")):
            q.update({f"Joint{i+1}_{side}": state[offset+i] for i in range(7)})
            q.update({f"Joint{i}_{side}": state[offset+7] / 2 for i in (8, 9)})
        links = {self.root: transform(BASE)}
        remaining = list(self.joints)
        while remaining:
            pending = []
            for name, kind, parent, child, origin, axis in remaining:
                if parent not in links:
                    pending.append((name, kind, parent, child, origin, axis))
                    continue
                motion = np.eye(4)
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * q[name]).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis / np.linalg.norm(axis) * q[name]
                elif kind != "fixed":
                    raise ValueError(f"Unsupported joint: {kind}")
                links[child] = links[parent] @ origin @ motion
            if len(pending) == len(remaining):
                raise ValueError("Disconnected/cyclic URDF")
            remaining = pending
        return links

    def skeleton(self, state):
        from matplotlib import colormaps
        links = self.forward(state)
        segments = [(links[p][:3, 3], links[c][:3, 3]) for _, _, p, c, _, _ in self.joints]
        # Origins alone can lie outside wrist views. Extend fingers to the
        # URDF's gripper-tip plane and clip, analogous to KinDER v2.
        for side in ("R", "L"):
            base = links[f"Pika_Gripper_Base_{side}"]
            inverse = np.linalg.inv(base)
            tip_z = (inverse @ links[f"Gripper_Tip_{side}"])[2, 3]
            for i in (8, 9):
                p = links[f"Link{i}_{side}"][:3, 3]
                local = inverse @ np.r_[p, 1.]
                local[2] = tip_z
                segments.append((p, (base @ local)[:3]))
        colors = (colormaps["tab20"].resampled(len(segments))(np.arange(len(segments)))[:, :3] * 255).astype(int)
        return {name: draw_skeleton(segments, colors, pose, fov)
                for name, (pose, fov) in camera_poses(links).items()}


def draw_skeleton(segments, colors, world_from_camera, fov, size=SIZE):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    extrinsic = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(world_from_camera)
    focal = 480 / (2 * np.tan(np.deg2rad(fov) / 2))
    for (start, end), color in zip(segments, colors):
        a, b = (np.c_[np.asarray([start, end]), np.ones(2)] @ extrinsic.T)[:, :3]
        near = 1e-4
        if max(a[2], b[2]) < near:
            continue
        if a[2] < near:
            a = a + (b-a) * ((near-a[2]) / (b[2]-a[2]))
        if b[2] < near:
            b = b + (a-b) * ((near-b[2]) / (a[2]-b[2]))
        points = np.asarray([a, b])
        uv = points[:, :2] / points[:, 2:] * focal + [424, 240]
        uv *= [size / 848, size / 480]
        u, v = np.rint(uv).clip(-1000000, 1000000).astype(int)
        visible, u, v = cv2.clipLine((0, 0, size, size), tuple(u), tuple(v))
        if visible:
            cv2.line(canvas, u, v, color.tolist(), 1, cv2.LINE_AA)
    return canvas


def pack(frames):
    panels = [cv2.resize(frames[c], (SIZE, SIZE), interpolation=cv2.INTER_AREA) for c in PANELS]
    return np.concatenate((np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)), axis=0)
