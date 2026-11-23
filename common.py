"""Shared provenance, atomic output, and existing repository backbones."""

# Small utilities used by every script:
#   * hashing helpers, so checkpoints record exactly which weights and data they came from;
#   * atomic file writes and an exclusive per-folder lock, so a crash or a second process cannot
#     leave a half-written or double-written result;
#   * seeding helpers for reproducible runs;
#   * construction and loading of the encoder backbones.

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

# Allow both package imports and running the scripts directly from the repository root.
if __package__:
    from . import benchmark_networks as networks
else:
    import benchmark_networks as networks


# Teacher seed used in the paper for each dataset.
TEACHER_SEEDS = {'CIFAR10': 0, 'CIFAR100': 0, 'TinyImageNet': 1}


def digest_json(value):
    """SHA-256 of a JSON-serializable value (stable: keys are sorted, NaN/inf are rejected)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    """SHA-256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Write `value` as indented JSON via a temporary file, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)  # atomic rename


def atomic_save(path, value):
    """`torch.save` through a temporary file and an atomic rename (safe against crashes mid-write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_run(folder):
    """OS lock releases on process exit, including crashes; no stale PID guessing."""
    import fcntl  # POSIX only, hence Linux/macOS (use WSL on Windows)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # non-blocking: fail fast if already locked
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is working in {folder}') from error
        yield


def seed_all(seed):
    """Seed Python, NumPy and PyTorch and force deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))  # NumPy only accepts seeds below 2**32
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id):
    """DataLoader `worker_init_fn`: derive Python/NumPy seeds from each worker's PyTorch seed."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    """Snapshot of the Python, PyTorch CPU and CUDA RNG states (stored in resumable checkpoints)."""
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    """Restore the RNG states captured by `rng_state`."""
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def state_hash(model):
    """SHA-256 over all parameters and buffers (incl. BatchNorm statistics), in a fixed key order."""
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_encoder(model, size):
    """Use the root BN networks; classifier is removed, projector is separate.

    Args:
        model: 'ResNet18', 'ResNet34', 'ResNet50', 'VGG11' or 'ConvNet'.
        size:  input resolution (matters only for ConvNet depth: 4 blocks at 64 px, otherwise 3).

    Returns:
        (encoder that outputs flat feature vectors, feature dimension).
    """
    if model in ('ResNet18', 'ResNet34', 'ResNet50'):
        if model == 'ResNet18':
            encoder = networks.ResNet18(channel=3, num_classes=1)
        else:
            # ResNet-34 uses basic blocks, ResNet-50 uses bottleneck blocks; both have stages [3, 4, 6, 3].
            block = networks.BasicBlock if model == 'ResNet34' else networks.Bottleneck
            encoder = networks.ResNet(block, [3, 4, 6, 3], channel=3, num_classes=1, norm='batchnorm')
        dimension = encoder.classifier.in_features
        # Standard residual initialization; no AP substitution.
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Zero-init the last BatchNorm of every residual branch so each block starts as the identity.
        for module in encoder.modules():
            if isinstance(module, networks.BasicBlock):
                nn.init.zeros_(module.bn2.weight)
            elif isinstance(module, networks.Bottleneck):
                nn.init.zeros_(module.bn3.weight)
    elif model == 'VGG11':
        encoder = networks.VGG11BN(channel=3, num_classes=1)
        dimension = encoder.classifier.in_features
    elif model == 'ConvNet':
        encoder = networks.ConvNet(3, 1, 128, 4 if size == 64 else 3,
                                   'relu', 'batchnorm', 'avgpooling', (size, size))
        dimension = encoder.classifier.in_features
    else:
        raise ValueError(model)
    encoder.classifier = nn.Identity()  # drop the dummy 1-class head: the encoder returns features
    if any(isinstance(module, (nn.GroupNorm, nn.InstanceNorm2d)) for module in encoder.modules()):
        raise RuntimeError('Benchmark accidentally selected a non-BN backbone')
    return encoder, dimension


def load_encoder(path, device='cuda'):
    """Load an encoder checkpoint written by ssl_training.train and return it frozen, in eval mode.

    Returns:
        (encoder, feature dimension, the full checkpoint dictionary).
    """
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'ssl_reference_encoder' or not saved.get('complete'):
        raise ValueError(f'Not a complete benchmark encoder: {path}')
    model, dimension = build_encoder(saved['model'], saved['input_size'])
    model.load_state_dict(saved['encoder'], strict=True)  # strict: any missing or extra key is an error
    if state_hash(model) != saved['encoder_sha256']:
        raise ValueError(f'Encoder state hash mismatch: {path}')
    return model.to(device).eval().requires_grad_(False), dimension, saved
