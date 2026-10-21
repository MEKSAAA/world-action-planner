"""Compare the same held-out windows before/after finetuning, including controls."""
import argparse
import json
from pathlib import Path


def compare(baseline, finetuned):
    a, b = ({r['window']: r for r in rows} for rows in (baseline, finetuned))
    if len(a) != len(baseline) or len(b) != len(finetuned) or not a or a.keys() != b.keys():
        raise ValueError("Require identical, nonempty, unique held-out window sets")
    for key in a:
        for field in ('window_sha256', 'seed', 'episode_index'):
            if a[key][field] != b[key][field]:
                raise ValueError(f"Comparison mismatch: {key} {field}")
    lines = ['| View | Metric | Baseline true | Finetuned true | Finetuned frozen | Copy last |',
             '|---|---|---:|---:|---:|---:|']
    for view in ('ego', 'wrist_right', 'wrist_left'):
        for metric in ('psnr', 'ssim', 'robot_region_mse', 'changed_region_mse', 'lpips'):
            values = []
            for rows, mode in ((a, 'true'), (b, 'true'), (b, 'frozen'), (a, 'copy_last')):
                v = [r[mode][view].get(metric) for r in rows.values()]
                v = [x for x in v if x is not None]
                values.append(f'{sum(v)/len(v):.4f}' if v else '—')
            if values != ['—']*4:
                lines.append('| ' + ' | '.join([view, metric, *values]) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--finetuned', type=Path, required=True)
    args = p.parse_args()
    print(compare(json.loads(args.baseline.read_text()), json.loads(args.finetuned.read_text())))


if __name__ == '__main__':
    main()
