"""Paired true/frozen-condition WAP probe with copy-last and region baselines.

Uses the existing in-process WAP pipeline so both controls use the same RNG
seed. Only ground-truth future *robot* conditions enter prediction; future RGB
is used exclusively for scoring. No object poses or future RGB enter the model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIEW_SLOTS = {"ego": (0, 0), "wrist_right": (1, 0), "wrist_left": (1, 1)}


def view(video, slot):
    y, x = slot
    return video[:, y*224:(y+1)*224, x*224:(x+1)*224]


def score(pred, gt, cond, last, lpips=None):
    from skimage.metrics import structural_similarity
    if pred.shape != gt.shape or pred.shape != cond.shape:
        raise ValueError("Prediction/GT/condition lengths and dimensions must match")
    result = {}
    for name, slot in VIEW_SLOTS.items():
        p, g, c = (view(a, slot) for a in (pred, gt, cond))
        baseline = view(last[None], slot)[0]
        squared = (p.astype(float)-g.astype(float))**2
        mse_t = squared.mean(axis=(1, 2, 3))
        # Treat near-black codec residuals as background, not skeleton pixels.
        masks = [cv2.dilate((f.max(-1) > 24).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) for f in c]
        robot = [float(e[m].mean()) for e, m in zip(squared, masks) if m.any()]
        # A visual-change region, not an object/contact mask (also includes shadows).
        motion = np.abs(g.astype(float)-baseline).mean(-1) > 12
        changed = [float(e[m].mean()) for e, m in zip(squared, motion) if m.any()]
        psnr = 10*np.log10(255**2 / np.maximum(mse_t, 1e-12))
        res = {"mse": float(mse_t.mean()), "psnr": float(psnr.mean()),
               "ssim": float(np.mean([structural_similarity(a, b, channel_axis=-1, data_range=255) for a, b in zip(g, p)])),
               "robot_region_mse": float(np.mean(robot)) if robot else None,
               "changed_region_mse": float(np.mean(changed)) if changed else None,
               "region_frames": len(robot), "changed_frames": len(changed),
               "psnr_t": psnr.tolist()}
        if lpips is not None:
            import torch
            with torch.inference_mode():
                # Batch in small chunks to avoid adding substantial probe VRAM.
                values = []
                for i in range(0, len(p), 4):
                    a = torch.from_numpy(p[i:i+4].copy()).permute(0, 3, 1, 2).float()/255
                    b = torch.from_numpy(g[i:i+4].copy()).permute(0, 3, 1, 2).float()/255
                    values.extend(lpips(a, b).flatten().tolist())
                res["lpips"] = float(np.mean(values))
        result[name] = res
    return result


def checkpoint_config(checkpoint):
    # Includes permanent checkpoints under <run>/checkpoints/keep/.
    for parent in checkpoint.resolve().parents:
        folder = parent / '.hydra'
        if (folder/'config.yaml').is_file() and (folder/'hydra.yaml').is_file():
            return folder/'config.yaml'
    raise FileNotFoundError(f"No .hydra/config.yaml + hydra.yaml above {checkpoint}")


def make_pipeline(checkpoint, wan, prompt, negative, device):
    sys.path.insert(0, str(ROOT/"world_model"))
    from pipeline import VideoPredictionPipeline
    import torch
    # Paths are explicit: released configs contain the original author's paths.
    overrides = [
        "algorithm.model.ckpt_path=" + str(wan.resolve()),
        "algorithm.vae.ckpt_path=" + str((wan/"Wan2.1_VAE.pth").resolve()),
        "algorithm.vae.compile=false", "algorithm.model.compile=false",
        "dataset.data_root=" + str(ROOT.resolve()),
        "dataset.metadata_path=__inference_no_metadata__.csv",
        "dataset.augmentation.random_flip=null",
        "dataset.augmentation.scale=[1.0,1.0]", "dataset.augmentation.ratio=[1.0,1.0]",
        "dataset.download.override_fps=20", "dataset.fps=20",
    ]
    pipeline = VideoPredictionPipeline(str(checkpoint_config(checkpoint)), str(checkpoint.resolve()),
                                       device=device, dtype=torch.bfloat16, overrides=overrides)
    # Pipeline.build_batch loads default embeddings with an empty record.
    # Bind explicit files on this instance instead of changing module globals.
    original = pipeline.dataset._load_prompt_embed
    negative_path = negative
    def load_prompt(record, negative=False):
        return original({"prompt_embed_path": str(prompt.resolve()),
                         "negative_prompt_embed_path": str(negative_path.resolve())}, negative=negative)
    pipeline.dataset._load_prompt_embed = load_prompt
    pipeline.algorithm.eval()
    return pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--wan", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--negative-prompt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    p.add_argument("--lpips", action="store_true", help="Optional pyiqa LPIPS, evaluated on CPU")
    args = p.parse_args()
    for path in (args.checkpoint, args.prompt, args.negative_prompt, args.windows/"index.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    index = json.loads((args.windows/"index.json").read_text())
    if not index:
        p.error("Window index is empty")
    if any(row.get('fps') != 20 or row.get('cams') != ['ego', 'ego', 'wrist_right', 'wrist_left'] for row in index):
        p.error("Expected 20 FPS RoboWits panels: ego, ego, wrist_right, wrist_left")
    if args.limit is not None and args.limit < 1:
        p.error("limit must be positive")
    index = index[:args.limit]
    args.out.mkdir(parents=True, exist_ok=False)
    pipeline = make_pipeline(args.checkpoint, args.wan, args.prompt, args.negative_prompt, args.device)
    import torch
    import imageio.v2 as imageio
    lpips = None
    if args.lpips:
        import pyiqa
        lpips = pyiqa.create_metric("lpips", device="cpu")
    results = []
    for item in index:
        file = args.windows/(item['window']+'.npz')
        with np.load(file) as z:
            history, past, gt, cond = (z[k] for k in ("hist_rgb", "hist_cond", "fut_rgb", "fut_cond"))
        row = {**item, "seed": args.seed, "window_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        copy_last = np.repeat(history[-1:], len(gt), axis=0)
        row["copy_last"] = score(copy_last, gt, cond, history[-1], lpips)
        predictions = {}
        for mode in ("true", "frozen"):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            sent = cond if mode == "true" else np.repeat(past[-1:], len(cond), axis=0)
            started = time.monotonic()
            with torch.inference_mode():
                output = pipeline(list(history), list(past), list(sent), return_type="np")
            pred = np.stack(output.pred_frames)
            if pred.shape != gt.shape:
                raise ValueError(f"No frame interpolation allowed: prediction {pred.shape}, GT {gt.shape}")
            row[mode] = score(pred, gt, cond, history[-1], lpips)
            row[mode+"_seconds"] = time.monotonic()-started
            predictions[mode] = pred
            print(f"{item['window']} {mode}: {row[mode]['ego']}", flush=True)
        np.savez_compressed(args.out/(item['window']+'_pred.npz'), **predictions)
        with imageio.get_writer(args.out/(item['window']+'.mp4'), fps=int(item['fps']), codec="libx264") as writer:
            for frames in zip(gt, cond, predictions['true'], predictions['frozen']):
                writer.append_data(np.concatenate(frames, axis=1))
        results.append(row)
        (args.out/"metrics.partial.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    (args.out/"metrics.partial.json").rename(args.out/"metrics.json")
    (args.out/"run.json").write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size, "seed": args.seed,
        "panels_scored": list(VIEW_SLOTS), "video_order": ["GT", "condition", "true", "frozen"],
        "windows": str(args.windows.resolve()), "pid": os.getpid()}, indent=2))


if __name__ == '__main__':
    main()
