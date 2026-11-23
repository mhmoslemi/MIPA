"""Train an SSL encoder from paired augmented views."""

# Used for three things: the frozen teacher (train_teacher.py), the student pretrained on synthetic or
# random images (eval.py), and the Barlow Twins variant. Training is deterministic and resumable:
# every epoch is seeded on its own, and a full checkpoint is written periodically.

import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

# Allow both package imports and running the scripts directly from the repository root.
if __package__:
    from .common import atomic_save, build_encoder, load_encoder, restore_rng, rng_state, seed_all, state_hash
    from .data import INPUT_SIZE, SSL_NORMALIZATION, ImageView, loader, ssl_transform
else:
    from common import atomic_save, build_encoder, load_encoder, restore_rng, rng_state, seed_all, state_hash
    from data import INPUT_SIZE, SSL_NORMALIZATION, ImageView, loader, ssl_transform


class LARS(torch.optim.Optimizer):
    """Momentum SGD with the clipped LARS rule used by the CIFAR reference.

All bias and normalization parameters are excluded from decay and adaptation.
The denominator uses ||gradient|| + decay*||weight|| + epsilon, before momentum.
"""
    def __init__(self, parameters, lr):
        super().__init__(parameters, dict(lr=lr, momentum=0.9, weight_decay=1e-4,
                                         eta=0.02, eps=1e-8, clip=True))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                # Weight matrices and conv kernels (ndim > 1) get weight decay and a layer-wise trust
                # ratio; biases and norm parameters (ndim == 1) are updated by plain momentum SGD.
                if parameter.ndim > 1:
                    weight_norm, grad_norm = parameter.norm(), update.norm()
                    trust = group['eta'] * weight_norm / (
                        grad_norm + group['weight_decay'] * weight_norm + group['eps'])
                    trust = torch.where((weight_norm > 0) & (grad_norm > 0), trust,
                                        torch.ones_like(trust))
                    if group['clip']:
                        # "clipped" LARS: the effective scale never exceeds the base learning rate
                        trust = torch.clamp(trust / max(group['lr'], 1e-16), max=1.0)
                    update = (update + group['weight_decay'] * parameter) * trust
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(parameter)
                velocity = state['momentum_buffer']
                velocity.mul_(group['momentum']).add_(update)
                parameter.add_(velocity, alpha=-group['lr'])
        return loss


def cosine_lr(step, total, peak, warmup=0, start=0., minimum=0.):
    """Learning rate at `step`: linear warmup from `start` to `peak`, then cosine decay to `minimum`."""
    if warmup and step < warmup:
        return start + (peak - start) * step / max(1, warmup - 1)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def simclr_loss(first, second, temperature):
    """NT-Xent (SimCLR) loss for two aligned batches of projections of shape (B, q).

    Each of the 2B projections must pick its other view out of the remaining 2B - 1 candidates.
    """
    if len(first) != len(second) or len(first) < 2:
        raise ValueError('SimCLR requires paired batches with at least two images')
    features = F.normalize(torch.cat((first, second)).float(), dim=1)
    scores = features @ features.T / temperature  # (2B, 2B) cosine similarities
    scores.fill_diagonal_(-torch.inf)  # an image is never its own candidate
    count = len(first)
    # row i's positive is the other view of the same image: i + B (first half) or i - B (second half)
    positives = (torch.arange(2 * count, device=scores.device) + count) % (2 * count)
    return F.cross_entropy(scores, positives)


def barlow_loss(first, second, lamb=0.0051, scale=0.1, eps=1e-5):
    """Barlow Twins loss: push the cross-correlation matrix of the two views toward the identity.

    Args:
        first, second: (B, q) projections of the two views.
        lamb:  weight of the off-diagonal (redundancy) term.
        scale: overall loss scale.
    """
    if first.shape != second.shape or len(first) < 2:
        raise ValueError('Barlow requires paired batches with at least two images')
    # Biased (divide-by-N) variance matches BatchNorm1d(affine=False).
    first, second = first.float(), second.float()
    # standardize every feature dimension across the batch
    a = (first - first.mean(0)) * torch.rsqrt(first.var(0, unbiased=False) + eps)
    b = (second - second.mean(0)) * torch.rsqrt(second.var(0, unbiased=False) + eps)
    correlation = a.T @ b / len(a)  # (q, q) cross-correlation between the two views
    diagonal = torch.diagonal(correlation)
    dimension = correlation.shape[0]
    # view the flattened matrix as (q-1, q+1) rows so that dropping the first column removes the diagonal
    off = correlation.flatten()[:-1].view(dimension - 1, dimension + 1)[:, 1:].flatten()
    return scale * ((diagonal - 1).square().sum() + lamb * off.square().sum())


class SSLModel(nn.Module):
    """Encoder plus projection head. Only the encoder is kept after pretraining."""

    def __init__(self, model, size, method):
        super().__init__()
        self.encoder, self.dimension = build_encoder(model, size)
        barlow = method == 'barlowtwins'
        # SimCLR: d -> 2048 -> 256.  Barlow Twins: d -> 2048 -> 2048 -> 2048 (with BatchNorm between layers).
        dims = [self.dimension, 2048, 2048, 2048] if barlow else [self.dimension, 2048, 256]
        layers = []
        for index, (incoming, outgoing) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(incoming, outgoing))
            if index < len(dims) - 2:  # no activation after the last layer
                if barlow:
                    layers.append(nn.BatchNorm1d(outgoing))
                layers.append(nn.ReLU())
        self.projector = nn.Sequential(*layers)

    def forward(self, first, second):
        """Project both views of a batch; returns two (B, q) tensors."""
        return self.projector(self.encoder(first)), self.projector(self.encoder(second))


def matches_run(saved, identity, output):
    """True if a saved checkpoint belongs to the run described by `identity` (current or legacy format)."""
    if saved['identity'] == identity:
        return True
    metadata = output / 'config.json'
    if 'signature' not in saved['identity'] or not metadata.is_file():
        return False
    # Older checkpoints kept their run settings beside the encoder.
    config = json.loads(metadata.read_text())
    if config['identity'] != saved['identity']:
        return False
    previous = config['signature']
    ssl = previous['protocol']['ssl']
    epoch_key = 'full_epochs' if config['identity']['selection_method'] == 'full' else 'subset_epochs'
    fields = {key: previous[key] for key in ('source_fingerprint', 'image_source') if key in previous}
    return identity == dict(source=previous['source'], model=previous['model'], method=previous['ssl'],
                            seed=config['identity']['seed'], epochs=ssl[epoch_key],
                            batch_size=ssl['batch_size'], **fields)


def train(dataset, source, model_name, method, seed, output, inputs, *,
          epochs, batch_size, workers, device, checkpoint_every):
    """Pretrain an encoder with SimCLR or Barlow Twins and save it under `output`.

    Args:
        dataset:    images to train on (real data, a random subset, or synthetic images).
        source:     dataset name; selects image size and augmentation statistics.
        model_name: backbone name understood by common.build_encoder.
        method:     'simclr' or 'barlowtwins'.
        seed:       training seed.
        output:     folder for encoder.pt, the resumable training.pt and the per-epoch ssl.jsonl log.
        inputs:     extra provenance entries recorded in the run identity (data fingerprints, ...).
        epochs, batch_size, workers, device: usual training settings.
        checkpoint_every: write a resumable checkpoint every this many epochs.

    Returns:
        (path of encoder.pt, checkpoint metadata dictionary). If a matching finished encoder already
        exists it is returned immediately; a mismatching one raises an error.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Everything that defines the run; used to decide whether existing files may be reused or resumed.
    identity = dict(source=source, model=model_name, method=method, seed=seed,
                    epochs=epochs, batch_size=batch_size, **inputs)
    destination = output / 'encoder.pt'
    if destination.exists():
        encoder, _, saved = load_encoder(destination, device='cpu')
        if not matches_run(saved, identity, output):
            raise ValueError(f'An encoder for a different run exists in {output}. Choose a new --output folder.')
        del encoder
        return destination, saved
    seed_all(seed + 3000)
    size = INPUT_SIZE[source]
    effective_batch = min(len(dataset), batch_size)  # tiny synthetic sets can be smaller than the batch size
    if effective_batch < 2:
        raise ValueError('Cannot train SSL on fewer than two distinct images')
    stream = loader(ImageView(dataset, ssl_transform(source, method)),
                    effective_batch, workers, seed, shuffle=True, drop_last=True)
    model = SSLModel(model_name, size, method).to(device)
    # Peak learning rate (scaled down linearly if the batch had to be shrunk) and LARS optimizer.
    peak = (0.4 if method == 'simclr' else 0.3) * (effective_batch / batch_size)
    optimizer = LARS(model.parameters(), peak)
    total_steps = epochs * len(stream)
    warmup_steps = min(10, max(0, epochs - 1)) * len(stream)  # up to 10 warmup epochs
    resume = output / 'training.pt'
    start_epoch = steps = 0
    if resume.exists():
        # Continue an interrupted run from the last periodic checkpoint.
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if not matches_run(saved, identity, output) or saved['steps_per_epoch'] != len(stream):
            raise ValueError('Cannot resume with these inputs or settings. Choose a new --output folder.')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, steps = saved['epoch'], saved['steps']
        restore_rng(saved['rng'])
    with (output / 'ssl.jsonl').open('a', buffering=1) as log:
        for epoch in range(start_epoch, epochs):
            # Seed each epoch so interrupted and uninterrupted runs sample the same views.
            epoch_seed = seed * 1000003 + epoch + 10000
            seed_all(epoch_seed)
            stream.generator.manual_seed(epoch_seed)
            model.train()
            start_time = time.monotonic()
            weighted_loss = examples = 0
            for (first, second), _unused_labels in stream:
                # Per-step learning rate: linear warmup from 3e-5, then cosine decay to zero.
                lr = cosine_lr(steps, total_steps, peak, warmup_steps, start=3e-5)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                first, second = first.to(device, non_blocking=True), second.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                z1, z2 = model(first, second)
                loss = simclr_loss(z1, z2, temperature=0.2) if method == 'simclr' else barlow_loss(z1, z2)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite SSL loss at epoch {epoch}, step {steps}')
                loss.backward()
                finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                if not finite:
                    raise RuntimeError(f'Non-finite SSL gradient at epoch {epoch}, step {steps}')
                optimizer.step()
                steps += 1
                examples += len(first)
                weighted_loss += float(loss.detach()) * len(first)
            # One log line per epoch; projector_std near zero would indicate representation collapse.
            stats = {'epoch': epoch + 1, 'steps': steps, 'lr': lr,
                     'loss': weighted_loss / examples, 'examples': examples,
                     'projector_std': float(z1.detach().std(0, unbiased=False).mean()),
                     'seconds': time.monotonic() - start_time}
            line = json.dumps(stats, allow_nan=False)
            print(line, file=log)
            print(f'SSL {source}/{model_name}/{method} seed={seed}: {line}', flush=True)
            if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
                atomic_save(resume, {'identity': identity, 'epoch': epoch + 1, 'steps': steps,
                                    'steps_per_epoch': len(stream), 'model': model.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'rng': rng_state()})
    # Keep only the encoder (the projection head is discarded) and save it with its provenance.
    encoder = model.encoder.cpu()
    saved = {'kind': 'ssl_reference_encoder', 'complete': True, 'identity': identity,
             'model': model_name, 'input_size': size, 'source': source, 'method': method,
             'seed': seed, 'epochs': epochs, 'steps': steps, 'effective_batch': effective_batch,
             'encoder': encoder.state_dict(), 'encoder_sha256': state_hash(encoder),
             'normalization': SSL_NORMALIZATION[source]}
    atomic_save(destination, saved)
    # Reload strictly before any probe: no missing keys, classifier mismatch,
    # silent random initialization, or projector features can pass this boundary.
    checked, _, _ = load_encoder(destination, device='cpu')
    del checked
    return destination, {key: value for key, value in saved.items() if key != 'encoder'}
