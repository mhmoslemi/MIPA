import random

import torch
import torch.nn as nn

if __package__:
    from .utils import DiffAugment, ParamDiffAug
else:
    from utils import DiffAugment, ParamDiffAug


def make_diff_augmenter(strategy, param):
    """Use a shared seed for single-operation DSA on normalized images."""
    param.aug_mode = 'S'

    def augment(images, seed):
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            return DiffAugment(images, strategy, seed=int(seed), param=param)

    return augment


class DMDSA(nn.Module):
    """Apply DSA in normalized space and return RGB pixels."""

    def __init__(self, strategy, mean, std):
        super().__init__()
        self.strategy = strategy
        self.param = ParamDiffAug()
        self.register_buffer('mean', torch.as_tensor(mean).reshape(1, -1, 1, 1).clone())
        self.register_buffer('std', torch.as_tensor(std).reshape(1, -1, 1, 1).clone())

    @staticmethod
    def seeds(count):
        return [random.randrange(100000) for _ in range(count)]

    def normalized(self, images, seed=None):
        if seed is None:
            seed = self.seeds(1)[0]
        return DiffAugment(images, self.strategy, seed=seed, param=self.param)

    def forward(self, pixels, seed=None):
        images = self.normalized((pixels - self.mean) / self.std, seed)
        return images * self.std + self.mean

    def configuration(self):
        return {'backend': 'utils.DiffAugment', 'strategy': self.strategy,
                'parameters': vars(ParamDiffAug()), 'input_space': 'dataset_normalized',
                'real_synthetic_coupling': 'shared_seed_per_view'}
