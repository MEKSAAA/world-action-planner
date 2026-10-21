"""Build WAP paired videos from measured RoboWits demonstrations (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from .data import Episodes, VideoWriter, decode_frames, sample_indices
from .geometry import CAMERAS, PANELS, Marvin, pack


def build_episode(source, rec, robot, out, fps, pad, prompt, negative):
    episode = int(rec["episode_index"])
    name = f"ep{episode:06d}"
    folder = out / name
    folder.mkdir()  # Fail on a partial previous run too: do not silently mix data.
    state, action = source.states(rec)
    indices = sample_indices(len(state), source.fps, fps)
    rgb_streams = {}
    for cam in CAMERAS:
        path, offset = source.video(rec, cam)
        rgb_streams[cam] = decode_frames(path, offset + indices / source.fps, source.fps)
    video, pose = VideoWriter(folder/"rgb.mp4", fps), VideoWriter(folder/"pose.mp4", fps)
    coverage = {c: [] for c in CAMERAS}
    try:
        for i, source_frame in enumerate(indices):
            rgb = pack({c: next(rgb_streams[c]) for c in CAMERAS})
            skeleton = robot.skeleton(state[source_frame])
            cond = pack(skeleton)
            for c in CAMERAS:
                coverage[c].append(int(np.any(skeleton[c] > 0, axis=-1).sum()))
            for _ in range(pad+1 if i == 0 else 1):
                video.write(rgb)
                pose.write(cond)
            if i in {0, len(indices)//2, len(indices)-1}:
                overlay = rgb.copy()
                mask = cond.max(axis=-1) > 0
                overlay[mask] = cond[mask]
                cv2.imwrite(str(folder/f"overlay_{i:05d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        video.close()
        pose.close()
        for reader in rgb_streams.values():
            reader.close()
    # Preserve the exact mapping (including warmup) to the original source rows.
    frame_map = np.r_[np.zeros(pad, dtype=int), indices]
    np.save(folder/"source_indices.npy", frame_map)
    result = {
        "video_path": f"{name}/rgb.mp4", "cond_path": f"{name}/pose.mp4",
        "fps": fps, "n_frames": len(frame_map), "width": 448, "height": 448,
        "prompt_embed_path": str(prompt), "negative_prompt_embed_path": str(negative),
        "split": rec["split"], "episode_index": episode, "task_index": int(rec["task_index"]),
        "instruction": rec["instruction"], "pad": pad, "cams": "|".join(PANELS),
        "source_root": str(source.root), "source_fps": source.fps,
        "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(robot.urdf.read_bytes()).hexdigest(),
        "condition_source": "measured_joint_state_same_frame; zero_physics_steps",
        "coverage": {c: {"min_pixels": min(v), "mean_pixels": float(np.mean(v)),
                         "empty_frames": v.count(0)} for c, v in coverage.items()},
    }
    (folder/"episode.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--negative-prompt", type=Path, required=True)
    p.add_argument("--fps", type=int, choices=[20], default=20)
    p.add_argument("--pad", type=int, default=60)
    p.add_argument("--holdout-per-task", type=int, default=3)
    p.add_argument("--validation-per-task", type=int, default=2)
    p.add_argument("--task-indices", type=int, nargs="+")
    p.add_argument("--episode-ids", type=int, nargs="+")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse complete episodes only when the manifest and source hashes match")
    args = p.parse_args()
    if args.pad < 0 or args.fps < 1:
        p.error("pad must be nonnegative; fps must be positive")
    source = Episodes(args.source)
    # Split on the full source inventory before applying any subset selection.
    records = source.records(args.holdout_per_task, args.validation_per_task)
    selected = [r for r in records if (args.task_indices is None or r["task_index"] in args.task_indices)
                and (args.episode_ids is None or r["episode_index"] in args.episode_ids)]
    if not selected:
        p.error("No matching episodes")
    if args.episode_ids and set(args.episode_ids) != {r['episode_index'] for r in selected}:
        p.error("Some requested episodes do not exist or conflict with task selection")
    plan = {"source": str(source.root), "fps": args.fps, "pad": args.pad,
            "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
            "source_info_sha256": hashlib.sha256((source.root/'meta/info.json').read_bytes()).hexdigest(),
            "renderer_sha256": hashlib.sha256(Path(__file__).with_name('geometry.py').read_bytes()).hexdigest(),
            "prompt": str(args.prompt.resolve()), "negative_prompt": str(args.negative_prompt.resolve()),
            "reference": "RobertCHU01/world-action-planner@e7f1345",
            "robowits_reference": "hierarchical-subgoal-genesis131@f5d68f2",
            "panels": PANELS, "selected_episode_ids": [int(r['episode_index']) for r in selected],
            "splits": [{"episode_index": int(r['episode_index']), "task_index": int(r['task_index']),
                        "split": r['split']} for r in records]}
    if args.resume:
        if json.loads((args.out/"manifest.json").read_text()) != json.loads(json.dumps(plan)):
            raise ValueError("Resume manifest differs; use a fresh output directory")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out/"manifest.json").write_text(json.dumps(plan, indent=2))
    if args.plan_only:
        print(f"Planned {len(selected)} episodes in {args.out}")
        return
    for path in (args.prompt, args.negative_prompt):
        if not path.is_file():
            raise FileNotFoundError(path)
    robot = Marvin(args.urdf)
    rows = []
    for r in selected:
        cached = args.out/f"ep{int(r['episode_index']):06d}"/"episode.json"
        if args.resume and cached.is_file():
            import av
            row = json.loads(cached.read_text())
            state, action = source.states(r)
            if row['state_sha256'] != hashlib.sha256(state.tobytes()).hexdigest() or row['action_sha256'] != hashlib.sha256(action.tobytes()).hexdigest():
                raise ValueError(f"Source changed for {cached}")
            for key in ('video_path', 'cond_path'):
                with av.open(str(args.out/row[key])) as video:
                    if video.streams.video[0].frames != row['n_frames']:
                        raise ValueError(f"Incomplete cached video: {row[key]}")
        else:
            row = build_episode(source, r, robot, args.out, args.fps, args.pad,
                                args.prompt.resolve(), args.negative_prompt.resolve())
        rows.append({k: v for k, v in row.items() if k != "coverage"})
        pd.DataFrame(rows).to_csv(args.out/"metadata.partial.csv", index=False)
        print(f"episode {row['episode_index']}: {row['split']}, {row['n_frames']} frames", flush=True)
    (args.out/"metadata.partial.csv").rename(args.out/"metadata.csv")


if __name__ == "__main__":
    main()
