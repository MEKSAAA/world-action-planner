"""Copy numeric LeRobot data to EE_ABS while sharing read-only source videos."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import Marvin, BASE, transform


def ee14(fk, states):
    """Base-relative Gripper_Tip FK, native Rpos,Lpos,Rrot,Lrot,Rgrip,Lgrip."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 16 or not np.isfinite(states).all():
        raise ValueError('Expected finite (N,16) joint states/targets')
    n = len(states)
    q = {}
    for offset, side in ((0, 'R'), (8, 'L')):
        q.update({f'Joint{i+1}_{side}': states[:, offset+i] for i in range(7)})
        q.update({f'Joint{i}_{side}': states[:, offset+7]/2 for i in (8, 9)})
    links = {fk.root: np.broadcast_to(transform(BASE), (n, 4, 4))}
    pending = list(fk.joints)
    while pending:
        rest = []
        for name, kind, parent, child, origin, axis in pending:
            if parent not in links:
                rest.append((name, kind, parent, child, origin, axis))
                continue
            motion = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            if kind in ('revolute', 'continuous'):
                motion[:, :3, :3] = Rotation.from_rotvec(q[name][:, None]*axis/np.linalg.norm(axis)).as_matrix()
            elif kind == 'prismatic':
                motion[:, :3, 3] = q[name][:, None]*axis/np.linalg.norm(axis)
            elif kind != 'fixed':
                raise ValueError(kind)
            links[child] = links[parent] @ origin @ motion
        if len(rest) == len(pending):
            raise ValueError('Disconnected FK tree')
        pending = rest
    right, left = (links[f'Gripper_Tip_{side}'] for side in ('R', 'L'))
    return np.column_stack((right[:, :3, 3]-BASE, left[:, :3, 3]-BASE,
                            Rotation.from_matrix(right[:, :3, :3]).as_rotvec(),
                            Rotation.from_matrix(left[:, :3, :3]).as_rotvec(),
                            states[:, 7], states[:, 15])).astype(np.float32)


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    result = {k: getattr(a, k)(axis=0).tolist() for k in ('min', 'max', 'mean', 'std')}
    result['count'] = [len(a)]
    result.update({f'q{int(q*100):02d}': np.quantile(a, q, axis=0).tolist() for q in (.01, .10, .50, .90, .99)})
    return result


def main():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--stats-episodes', type=int, nargs='+', default=list(range(105, 150)))
    args = p.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'meta', out/'meta')
    (out/'videos').symlink_to(source/'videos', target_is_directory=True)
    fk = Marvin(args.urdf)
    features = ('observation.state', 'action')
    training = {k: [] for k in features}
    episode_stats = {}
    for path in sorted((source/'data').glob('**/*.parquet')):
        table = pq.read_table(path)
        episodes = np.asarray(table['episode_index'])
        for key in features:
            raw = np.array(table[key].to_pylist(), dtype=np.float32)
            converted = np.concatenate([ee14(fk, raw[i:i+4096]) for i in range(0, len(raw), 4096)])
            table = table.set_column(table.schema.get_field_index(key), key, pa.array(converted.tolist(), type=pa.list_(pa.float32())))
            training[key].append(converted[np.isin(episodes, args.stats_episodes)])
            for ep in np.unique(episodes):
                # Episodes may span parquet files; gather before computing statistics.
                episode_stats.setdefault(int(ep), {}).setdefault(key, []).append(converted[episodes == ep])
        target = out/path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(None), target)
        print(f'Converted {path.name}: {len(table)} frames', flush=True)
    global_stats = json.loads((out/'meta/stats.json').read_text())
    for key in features:
        global_stats[key] = stats(np.concatenate(training[key]))
    (out/'meta/stats.json').write_text(json.dumps(global_stats, indent=2))
    for path in (out/'meta/episodes').glob('**/*.parquet'):
        df = pd.read_parquet(path)
        for key in features:
            values = [stats(np.concatenate(episode_stats[int(ep)][key])) for ep in df['episode_index']]
            for statistic in values[0]:
                df[f'stats/{key}/{statistic}'] = [v[statistic] for v in values]
        df.to_parquet(path, index=False)
    info = json.loads((out/'meta/info.json').read_text())
    names = [f'{arm}.{kind}.{axis}' for kind in ('position', 'rotvec') for arm in ('right', 'left') for axis in ('x', 'y', 'z')]+['right.gripper_width', 'left.gripper_width']
    for key in features:
        info['features'][key].update(shape=[14], names=names)
    (out/'meta/info.json').write_text(json.dumps(info, indent=2))
    (out/'conversion.json').write_text(json.dumps({'source': str(source), 'urdf': str(args.urdf.resolve()),
        'stats_episodes': args.stats_episodes, 'format': 'EE_ABS/EE, base-relative Gripper_Tip, canonical rotation vector',
        'videos': 'unchanged, symlinked; timestamps and episode numbering preserved'}, indent=2))
    print('Complete:', out)


if __name__ == '__main__':
    main()
