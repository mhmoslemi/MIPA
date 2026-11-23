"""Frozen teacher used during distillation: loading and input-normalization bridging."""

import torch
import torch.nn as nn

# Allow both package imports and running the scripts directly from the repository root.
if __package__:
    from .common import load_encoder
else:
    from common import load_encoder


class NormalizedTeacher(nn.Module):
    """Wrap the teacher so it accepts images normalized the way the *synthetic* images are stored.

    The synthetic images live in one normalization space (mean/std `source`), while the teacher was
    trained on inputs normalized with its own statistics (mean/std `target`). This wrapper converts
    between the two, so gradients can flow from the teacher back into the synthetic pixels.
    """

    def __init__(self, network, source, target):
        super().__init__()
        self.network = network
        # source = (mean, std) of the synthetic images, target = (mean, std) the teacher expects.
        for name, values in zip(('source_mean', 'source_std', 'teacher_mean', 'teacher_std'), (*source, *target)):
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1))

    def embed(self, images):
        """Feature vectors of `images` (N, 3, H, W) given in the synthetic normalization space."""
        # un-normalize to [0, 1] pixels, then re-normalize with the teacher's own statistics
        return self.network((images * self.source_std + self.source_mean - self.teacher_mean) / self.teacher_std)

    def forward(self, images):
        return self.embed(images)


def load_teacher(path, dataset, normalization, size, device):
    """Load a teacher checkpoint and wrap it for use on synthetic-space images.

    Args:
        path:          encoder checkpoint produced by train_teacher.py.
        dataset:       dataset the teacher must have been trained on.
        normalization: (mean, std) used for the synthetic images of this dataset.
        size:          expected native input resolution of that dataset.
        device:        torch device to move the teacher to.

    Returns:
        (frozen teacher in eval mode, teacher training seed, SHA-256 of the teacher weights).
    """
    network, _, saved = load_encoder(path, device=device)
    if (saved['model'], saved['source'], saved['input_size']) != ('ResNet18', dataset, size):
        raise ValueError('Use a ResNet-18 teacher trained on the source dataset at its native resolution')
    teacher = NormalizedTeacher(network, normalization, saved['normalization']).to(device).eval()
    return teacher, saved['seed'], saved['encoder_sha256']
