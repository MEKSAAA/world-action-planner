"""Extract deterministic holdout probes from the same paired dataset as training."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import decode_frames


def boundaries(n_frames, pad, history, future, count):
    if min(history, future, count) < 1:
        raise ValueError("history, future and count must be positive")
    # Do not score warmup frames as future ground truth.
    lo, hi = max(history, pad+1), n_frames-future
    if lo > hi:
        raise ValueError("Episode is too short for a probe window")
    return np.unique(np.linspace(lo, hi, count).astype(int))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--history", type=int, default=80)
    p.add_argument("--future", type=int, default=40)
    p.add_argument("--windows-per-episode", type=int, default=3)
    args = p.parse_args()
    if args.future < 20 or args.future % 4:
        p.error("Future must be >=20 and divisible by 4 for the released checkpoint")
    rows = pd.read_csv(args.data/"metadata.csv")
    rows = rows[rows.split == "holdout"]
    if rows.empty:
        p.error("No holdout episodes")
    if not (rows.fps == 20).all():
        p.error("WAP RoboWits probes require prepared 20 FPS videos")
    manifest = json.loads((args.data/"manifest.json").read_text())
    held = {r['episode_index'] for r in manifest['splits'] if r['split'] == 'holdout'}
    if not set(rows.episode_index) <= held:
        p.error("Holdout metadata disagrees with split manifest")
    args.out.mkdir(parents=True, exist_ok=False)
    index = []
    for row in rows.to_dict("records"):
        for t in boundaries(int(row['n_frames']), int(row['pad']), args.history, args.future, args.windows_per_episode):
            times = np.arange(t-args.history, t+args.future) / row['fps']
            rgb = np.stack(list(decode_frames(args.data/row['video_path'], times, row['fps'])))
            cond = np.stack(list(decode_frames(args.data/row['cond_path'], times, row['fps'])))
            name = f"ep{row['episode_index']:06d}_t{t:05d}"
            np.savez_compressed(args.out/f"{name}.npz", hist_rgb=rgb[:args.history], hist_cond=cond[:args.history],
                                fut_rgb=rgb[args.history:], fut_cond=cond[args.history:])
            index.append({"window": name, "episode_index": row['episode_index'], "task_index": row['task_index'],
                          "t": int(t), "fps": row['fps'], "cams": row['cams'].split('|'),
                          "source_manifest": str((args.data/'manifest.json').resolve())})
    (args.out/"index.json").write_text(json.dumps(index, indent=2))
    print(f"Saved {len(index)} holdout windows")


if __name__ == '__main__':
    main()
