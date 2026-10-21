"""Read released LeRobot v3 episodes, including shared-video time offsets."""
import json
from pathlib import Path
import av
import numpy as np
import pandas as pd

from .geometry import CAMERAS


class Episodes:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("This reader supports LeRobot v3.0")
        self.fps = float(self.info["fps"])
        paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        self.meta = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        self.meta = self.meta.sort_values("episode_index")
        if self.meta.episode_index.duplicated().any():
            raise ValueError("Duplicate episode metadata")
        tasks = pd.read_parquet(self.root / "meta/tasks.parquet")
        self.tasks = {str(text): int(row.task_index) for text, row in tasks.iterrows()}

    def records(self, n_holdout=3, n_val=2):
        if n_holdout < 1 or n_val < 1:
            raise ValueError("Reserve at least one holdout and validation episode per task")
        counts = {}
        records = []
        for row in self.meta.to_dict("records"):
            if len(row["tasks"]) != 1:
                raise ValueError("Expected one task per episode")
            task = self.tasks[str(row["tasks"][0])]
            ordinal = counts.get(task, 0)
            counts[task] = ordinal + 1
            split = "holdout" if ordinal < n_holdout else "validation" if ordinal < n_holdout+n_val else "training"
            records.append({**row, "task_index": task, "instruction": str(row["tasks"][0]), "split": split})
        if any(n <= n_holdout+n_val for n in counts.values()):
            raise ValueError("Split leaves a task with no training episodes")
        return records

    def states(self, record):
        path = self.root / self.info["data_path"].format(
            chunk_index=int(record["data/chunk_index"]), file_index=int(record["data/file_index"]))
        df = pd.read_parquet(path, filters=[("episode_index", "=", int(record["episode_index"]))])
        df = df.sort_values("frame_index")
        n = int(record["length"])
        if len(df) != n or not np.array_equal(df.frame_index, np.arange(n)):
            raise ValueError("Incomplete/noncontiguous episode frame indices")
        if not np.allclose(df.timestamp, np.arange(n) / self.fps, atol=1e-4):
            raise ValueError("Nonuniform source timestamps")
        state, action = np.stack(df["observation.state"]), np.stack(df.action)
        if state.shape != (n, 16) or action.shape != state.shape or not np.isfinite(state).all():
            raise ValueError("Invalid state/action shape or nonfinite measured state")
        return state.astype(np.float64), action

    def video(self, record, camera):
        if camera not in CAMERAS:
            raise ValueError(camera)
        key = "observation.images." + camera
        prefix = "videos/" + key
        path = self.root / self.info["video_path"].format(video_key=key,
            chunk_index=int(record[prefix+"/chunk_index"]), file_index=int(record[prefix+"/file_index"]))
        return path, float(record[prefix+"/from_timestamp"])


def sample_indices(length, source_fps, target_fps):
    if length < 1 or not 0 < target_fps <= source_fps:
        raise ValueError("Require 0 < target FPS <= source FPS and nonempty episode")
    # No endpoint stretch: output frame j represents j / target_fps seconds.
    count = int(np.floor((length-1) * target_fps / source_fps)) + 1
    return np.rint(np.arange(count) * source_fps / target_fps).astype(int)


def decode_frames(path, times, fps):
    """Yield nearest presentation-timestamp frames; fail rather than pad gaps."""
    times = np.asarray(times, dtype=float)
    if not len(times) or np.any(np.diff(times) < 0):
        raise ValueError("Expected sorted nonempty target timestamps")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(max(0, int((times[0]-.5) / stream.time_base)), stream=stream, backward=True)
        i = 0
        for frame in container.decode(stream):
            timestamp = float(frame.pts * stream.time_base)
            while i < len(times) and timestamp >= times[i] - .5/fps - 1e-6:
                if abs(timestamp-times[i]) > .51/fps:
                    raise ValueError(f"Video/state timestamp mismatch at {times[i]}: {timestamp}")
                yield frame.to_ndarray(format="rgb24")
                i += 1
            if i == len(times):
                return
        raise ValueError(f"Video ended after {i}/{len(times)} requested frames: {path}")


class VideoWriter:
    def __init__(self, path, fps=20, crf=12):
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=int(fps))
        self.stream.width = self.stream.height = 448
        self.stream.pix_fmt = "yuv444p"
        self.stream.options = {"crf": str(crf), "preset": "fast"}

    def write(self, rgb):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
