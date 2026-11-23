"""Differentiable augmentation (DSA) helpers used to compare real and synthetic images."""

import random

import torch
import torch.nn as nn

# Allow both package imports and running the scripts directly from the repository root.
if __package__:
    from .utils import DiffAugment, ParamDiffAug
else:
    from utils import DiffAugment, ParamDiffAug


def make_diff_augmenter(strategy, param):
    """Use a shared seed for single-operation DSA on normalized images.

    Returns `augment(images, seed)`. The same `seed` always produces the same random transformation, so
    the real batch and the synthetic set can be augmented identically within one view. A given call draws
    ONE set of random parameters and applies it to every image in the batch ("Siamese" mode of DSA).
    The operations are differentiable, hence gradients reach the synthetic pixels through the augmentation.
    """
    param.aug_mode = 'S'  # 'S' = pick ONE operation from the strategy per call (single-operation mode)

    def augment(images, seed):
        # fork_rng restores the global RNG afterwards, so augmenting never disturbs other random streams.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            return DiffAugment(images, strategy, seed=int(seed), param=param)

    return augment


class DMDSA(nn.Module):
    """Apply DSA in normalized space and return RGB pixels."""

    def __init__(self, strategy, mean, std):
        super().__init__()
        self.strategy = strategy
        self.param = ParamDiffAug()
        # dataset mean/std as (1, C, 1, 1) buffers so they move with the module across devices
        self.register_buffer('mean', torch.as_tensor(mean).reshape(1, -1, 1, 1).clone())
        self.register_buffer('std', torch.as_tensor(std).reshape(1, -1, 1, 1).clone())

    @staticmethod
    def seeds(count):
        """`count` random augmentation seeds (one per view)."""
        return [random.randrange(100000) for _ in range(count)]

    def normalized(self, images, seed=None):
        """Augment images that are already in normalized space; a fresh seed is drawn if none is given."""
        if seed is None:
            seed = self.seeds(1)[0]
        return DiffAugment(images, self.strategy, seed=seed, param=self.param)

    def forward(self, pixels, seed=None):
        """Augment [0, 1] pixels: normalize, augment, then convert back to pixel space."""
        images = self.normalized((pixels - self.mean) / self.std, seed)
        return images * self.std + self.mean

    def configuration(self):
        """Description of this augmentation setup, recorded for provenance."""
        return {'backend': 'utils.DiffAugment', 'strategy': self.strategy,
                'parameters': vars(ParamDiffAug()), 'input_space': 'dataset_normalized',
                'real_synthetic_coupling': 'shared_seed_per_view'}
