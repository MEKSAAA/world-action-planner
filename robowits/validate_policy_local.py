"""Local held-out action prediction check; not an environment success metric."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('sim-root', 'checkpoint', 'data', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.sim_root/'third_party/lerobot/src'))
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = 'cuda'
    policy = PI05Policy.from_pretrained(args.checkpoint, config=cfg).to('cuda').eval()
    pre, _ = make_pre_post_processors(cfg, pretrained_path=str(args.checkpoint),
                                      preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    dataset = LeRobotDataset('local/lemon-ee', root=args.data, episodes=[103, 104],
                            delta_timestamps={'action': [i/30 for i in range(cfg.chunk_size)]}, video_backend='pyav')
    episodes = np.array(dataset.hf_dataset['episode_index'])
    selected = []
    for episode in (103, 104):
        available = np.flatnonzero(episodes == episode)
        selected.extend(available[np.linspace(0, len(available)-1, min(64, len(available))).round().astype(int)].tolist())
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, selected), batch_size=4, num_workers=2)
    sums = {103: [0., 0], 104: [0., 0]}
    with torch.inference_mode():
        for batch in loader:
            ids = batch['episode_index'].tolist()
            mask = (~batch['action_is_pad']).to('cuda')
            batch = pre(batch)
            prediction = policy.predict_action_chunk(batch)
            target = batch['action'][:, :prediction.shape[1], :prediction.shape[2]]
            if prediction.shape[-1] != 14 or target.shape != prediction.shape:
                raise ValueError('Expected matching 14D normalized action chunks')
            error = (prediction.float()-target.float()).square().mean(dim=-1)
            for i, episode in enumerate(ids):
                sums[episode][0] += float(error[i][mask[i]].sum().cpu())
                sums[episode][1] += int(mask[i].sum().cpu())
    result = {'checkpoint': str(args.checkpoint.resolve()), 'episodes': [103, 104],
              'samples': len(selected), 'seed': 0, 'padding_excluded': True,
              'metric': 'MSE of normalized 14D actions; NOT environment task success',
              'per_episode': {str(ep): {'normalized_action_mse': s/n, 'valid_action_frames': n} for ep, (s, n) in sums.items()},
              'normalized_action_mse': sum(s for s, _ in sums.values())/sum(n for _, n in sums.values())}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
