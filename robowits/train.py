"""Portable WAP finetuning launcher, following the KinDER e7f1345 recipe."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate_data(root):
    rows = list(csv.DictReader((root/"metadata.csv").open()))
    manifest = json.loads((root/"manifest.json").read_text())
    split_map = {int(r["episode_index"]): r["split"] for r in manifest["splits"]}
    seen = set()
    for row in rows:
        episode = int(row["episode_index"])
        if episode in seen or split_map.get(episode) != row["split"]:
            raise ValueError("Duplicate episode or split differs from source manifest")
        seen.add(episode)
        for field in ("video_path", "cond_path", "prompt_embed_path", "negative_prompt_embed_path"):
            if not (root/row[field]).is_file():
                raise FileNotFoundError(root/row[field])
        if int(row["fps"]) != 20 or int(row["n_frames"]) < 101:
            raise ValueError("Training/validation require >=101 frames at 20 FPS")
    if not {"training", "validation"} <= {r["split"] for r in rows}:
        raise ValueError("Need both training and validation episodes")
    return rows


def command(args):
    def option(k, v):
        return k + "=" + json.dumps(v)
    load = str(args.resume.resolve()) if args.resume else str(args.init.resolve()) + ":model"
    values = {
        "name": "robowits_wm_finetune", "wandb.mode": "offline",
        "wandb.project": "wap_robowits", "wandb.entity": args.wandb_entity, "load": load,
        "hydra.run.dir": str(args.run_dir.resolve()),
        "dataset.data_root": str(args.data.resolve()), "dataset.metadata_path": "metadata.csv",
        "algorithm.model.ckpt_path": str(args.wan.resolve()),
        "algorithm.vae.ckpt_path": str((args.wan/"Wan2.1_VAE.pth").resolve()),
        "algorithm.vae.compile": False,
        "algorithm.n_frames": 41, "algorithm.hist_len": 21, "+algorithm.max_frames": 81,
        "+algorithm.hist_steps": list(range(0, 61, 3)),
        "algorithm.diffusion_forcing.cond_mode": "concat",
        "algorithm.lr_scheduler.num_warmup_steps": args.warmup,
        "experiment.num_nodes": 1, "experiment.training.lr": args.lr,
        "experiment.training.batch_size": args.batch_size,
        "experiment.training.optim.accumulate_grad_batches": args.accum,
        "experiment.training.max_steps": args.steps,
        "experiment.training.data.num_workers": args.workers,
        "experiment.training.checkpointing.every_n_train_steps": args.save_every,
        "+experiment.training.checkpointing.filename": "latest-{step}",
        "+experiment.training.checkpointing.save_top_k": 1,
        "experiment.training.checkpointing.save_last": "link",
        "+experiment.training.checkpointing_keep.every_n_train_steps": args.keep_every,
        "+experiment.training.checkpointing_keep.save_top_k": -1,
        "+experiment.training.checkpointing_keep.filename": "{step}",
        "experiment.validation.val_every_n_step": args.val_every * args.accum,
        "experiment.validation.batch_size": 1,
        "experiment.validation.limit_batch": args.val_batches,
        "experiment.validation.data.num_workers": args.workers,
        "+experiment.validation.num_sanity_val_steps": args.sanity_batches,
    }
    if args.resume:
        values["algorithm.model.tuned_ckpt_path"] = str(args.resume.resolve())
    return [args.python, "main.py", "experiment=exp_video", "dataset=robowits", "algorithm=wan_at2v",
            *[option(k, v) for k, v in values.items()]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--init", type=Path, required=True, help="Released WAP checkpoint, weights-only initialization")
    p.add_argument("--resume", type=Path, help="Full Lightning state; mutually exclusive in effect with init")
    p.add_argument("--wan", type=Path, required=True, help="Wan directory with config.json and Wan2.1_VAE.pth")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpus", default="0")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "local"), help="Offline logging namespace; no login needed")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=250)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--keep-every", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=2)
    p.add_argument("--sanity-batches", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if min(args.batch_size, args.accum, args.steps, args.save_every, args.keep_every, args.val_every) < 1:
        p.error("Batch, accumulation, step and checkpoint intervals must be positive")
    if min(args.workers, args.val_batches, args.sanity_batches) < 0:
        p.error("Workers and validation counts cannot be negative")
    validate_data(args.data)
    for path in (args.resume or args.init, args.wan/"config.json", args.wan/"Wan2.1_VAE.pth"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.run_dir.exists():
        raise FileExistsError("Use a fresh run directory, including when resuming")
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    args.run_dir.mkdir(parents=True)
    (args.run_dir/"launch.json").write_text(json.dumps({
        "command": cmd, "cwd": str(ROOT/"world_model"), "gpus": args.gpus,
        "reference_commit": "e7f1345e4a8f665a17a9c24be78df7c805655a1e",
        "data_manifest": json.loads((args.data/"manifest.json").read_text()),
    }, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, cwd=ROOT/"world_model", env=env, check=True)


if __name__ == "__main__":
    main()
