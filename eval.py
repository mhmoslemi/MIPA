"""Pretrain on a synthetic set and evaluate frozen features on selected datasets."""

# Evaluation protocol (the one used for every number in the paper):
#   1. Train a fresh encoder with SSL (SimCLR or Barlow Twins) on the synthetic images ONLY,
#      or on a random real subset of the same size (--random-subset) as the baseline.
#   2. Freeze the encoder and train a linear probe on 1% / 5% of the labels of each target dataset.
#   3. Report test accuracy of the probe after its final epoch, averaged over several probe seeds.
#
# Results are written to <output>/summary.json (mean and std per target and label budget) and one JSON
# per individual probe run under <output>/probes/.

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import to_pil_image

# Allow both `python -m MIPA.eval` (package imports) and `python eval.py` (plain imports).
if __package__:
    from .common import atomic_json, exclusive_run, file_hash, load_encoder, seed_all, state_hash
    from .data import INPUT_SIZE, PAPER_TARGETS, SOURCES, SPECS, SYNTHETIC_NORMALIZATION, ImageView, RawDataset, label_indices, loader, locate, probe_transform
    from .ssl_training import cosine_lr, train
else:
    from common import atomic_json, exclusive_run, file_hash, load_encoder, seed_all, state_hash
    from data import INPUT_SIZE, PAPER_TARGETS, SOURCES, SPECS, SYNTHETIC_NORMALIZATION, ImageView, RawDataset, label_indices, loader, locate, probe_transform
    from ssl_training import cosine_lr, train


class SyntheticImages(Dataset):
    """Wrap a tensor of synthetic images as an unlabeled dataset.

    Images are returned as PIL images (with a dummy label of -1) so that the usual SSL augmentation
    pipeline, which expects PIL inputs, can be applied to them exactly as it is to real data.
    """

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return to_pil_image(self.images[index]), -1


def linear_probe(checkpoint, encoder_hash, source, target, train_set, test_set,
                 percentage, seed, output, epochs, workers, device):
    """Train and evaluate one linear probe on top of a frozen encoder.

    Args:
        checkpoint:    path of the pretrained encoder checkpoint.
        encoder_hash:  SHA-256 of the encoder weights, used to validate cached results.
        source, target: names of the pretraining dataset and of the dataset being probed.
        train_set, test_set: RawDataset objects of the target (labels are used only here).
        percentage:    percentage of the target training labels the probe may use.
        seed:          probe seed; controls the labeled subset, head initialization and data order.
        output:        JSON file for the result (a finished, matching result is reused, not recomputed).
        epochs, workers, device: probe epochs, DataLoader workers and torch device.

    Returns:
        The result dictionary that was also written to `output`.
    """
    output = Path(output)
    indices = label_indices(train_set, percentage, seed)  # class-balanced labeled subset
    # Everything that defines this probe; a cached result is reused only if all of it matches.
    expected = {'source': source, 'target': target, 'seed': seed,
                'label_percentage': percentage, 'labeled_indices': indices,
                'train_fingerprint': train_set.fingerprint, 'test_fingerprint': test_set.fingerprint}
    if output.exists():
        result = json.loads(output.read_text())
        previous = result.get('expected', {})
        previous_epochs = previous.get('epochs', previous.get('evaluation', {}).get('epochs'))
        if (result.get('complete') and result.get('encoder_sha256') == encoder_hash and previous_epochs == epochs
                and all(previous.get(key) == value for key, value in expected.items())):
            return result
    expected['epochs'] = epochs

    # Frozen encoder + a freshly initialized linear head on its features.
    encoder, dimension, saved = load_encoder(checkpoint, device)
    initial_hash = state_hash(encoder)
    seed_all(seed + 6000)
    head = nn.Linear(dimension, train_set.classes).to(device)
    nn.init.normal_(head.weight, std=0.01)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1, momentum=0.9)
    selected = Subset(train_set, indices)
    train_loader = loader(ImageView(selected, probe_transform(source, True)),
                          min(256, len(indices)), workers, seed, shuffle=True)
    test_loader = loader(ImageView(test_set, probe_transform(source, False)), 256, workers, seed)
    total_steps = len(train_loader) * epochs
    step = 0
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)

    # --- Training: only the linear head learns; features come from the frozen encoder -------------------
    with output.with_suffix('.jsonl').open('w', buffering=1) as log:
        for epoch in range(epochs):
            epoch_seed = seed * 1000003 + epoch + 60000
            seed_all(epoch_seed)  # per-epoch seeding makes the data order reproducible
            train_loader.generator.manual_seed(epoch_seed)
            # Never call train() on a container that contains the backbone.
            encoder.eval()
            head.train()
            loss_sum = seen = correct = 0
            for images, labels in train_loader:
                lr = cosine_lr(step, total_steps, 0.1)  # cosine decay of the probe learning rate
                optimizer.param_groups[0]['lr'] = lr
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.no_grad():
                    features = encoder(images)
                logits = head(features)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite linear probe loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                seen += len(labels)
                loss_sum += float(loss.detach()) * len(labels)
                correct += int((logits.argmax(1) == labels).sum())
                step += 1
            stats = {'epoch': epoch + 1, 'lr': lr, 'loss': loss_sum / seen,
                     'train_accuracy_percent': 100 * correct / seen}
            print(json.dumps(stats, allow_nan=False), file=log)
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                print(f'PROBE {target} lbl={percentage} seed={seed}: {stats}', flush=True)

    # --- Testing ----------------------------------------------------------------------------------------
    # Evaluate once, at the prespecified final epoch. Never pick the best test epoch.
    head.eval()
    correct = total = 0
    per_class_correct = np.zeros(train_set.classes, dtype=np.int64)
    per_class_count = np.zeros_like(per_class_correct)
    feature_sum = feature_square_sum = None  # running sums for feature mean/variance diagnostics
    with torch.no_grad():
        for images, labels in test_loader:
            features = encoder(images.to(device, non_blocking=True))
            if not torch.isfinite(features).all():
                raise RuntimeError('Non-finite encoder features')
            predicted = head(features).argmax(1).cpu()
            hits = (predicted == labels).numpy()
            correct += int(hits.sum())
            total += len(labels)
            np.add.at(per_class_count, labels.numpy(), 1)
            np.add.at(per_class_correct, labels.numpy(), hits.astype(np.int64))
            sums, squares = features.double().sum(0), features.double().square().sum(0)
            feature_sum = sums if feature_sum is None else feature_sum + sums
            feature_square_sum = squares if feature_square_sum is None else feature_square_sum + squares
    # The encoder must be bit-identical before and after probing (weights and BatchNorm buffers).
    final_hash = state_hash(encoder)
    if final_hash != initial_hash or final_hash != saved['encoder_sha256']:
        raise RuntimeError('Encoder parameters or BatchNorm buffers changed during probing')
    present = per_class_count > 0
    label_counts = np.bincount(train_set.targets[indices], minlength=train_set.classes)
    variance = (feature_square_sum / total - (feature_sum / total).square()).clamp_min(0)
    result = {'complete': True, 'expected': expected, 'encoder_sha256': initial_hash,
              'encoder_unchanged': True, 'accuracy_percent': 100 * correct / total,
              'macro_accuracy_percent': float(100 * (per_class_correct[present] / per_class_count[present]).mean()),
              'labeled_count': len(indices), 'labeled_class_counts': label_counts.tolist(),
              'labeled_classes_present': int((label_counts > 0).sum()),
              'total_classes': train_set.classes, 'test_count': total,
              'mean_feature_std': float(variance.sqrt().mean()),
              'near_constant_feature_fraction': float((variance < 1e-10).float().mean()),
              'seconds': time.monotonic() - started}
    atomic_json(output, result)
    print(f'RESULT {target} lbl={percentage} seed={seed}: {result["accuracy_percent"]:.4f}', flush=True)
    return result


def main(argv=None):
    """Pretrain on the chosen images, then run every requested linear probe and write summary.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Exactly one image source: a distilled checkpoint or a random real subset of the same size.
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--synthetic-checkpoint', type=Path)
    inputs.add_argument('--random-subset', action='store_true')
    parser.add_argument('--dataset', choices=SOURCES, help='Required when the input has no dataset metadata')
    parser.add_argument('--percentage', type=float, help='Required when the input has no size metadata')
    parser.add_argument('--data-path', default='data')
    parser.add_argument('--targets', nargs='+', choices=('source', 'all', *SPECS), default=['source'],
                        help='Use all for the source and all paper transfer datasets')
    parser.add_argument('--labels', '--label-percentage', nargs='+', type=float, default=[1, 5])
    parser.add_argument('--model', type=lambda name: name.removesuffix('BN'),
                        choices=('ResNet18', 'ResNet34', 'ResNet50', 'ConvNet', 'VGG11'), default='ResNet18')
    parser.add_argument('--ssl-method', choices=('simclr', 'barlowtwins'), default='simclr')
    parser.add_argument('--ssl-epochs', type=int)
    parser.add_argument('--probe-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256, help='SSL batch size')
    parser.add_argument('--seed', type=int, default=0, help='SSL encoder seed')
    parser.add_argument('--probe-seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--output', '--save-path', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    # Paper defaults: SimCLR is pretrained for 2,000 epochs, Barlow Twins for 10,000.
    ssl_epochs = args.ssl_epochs if args.ssl_epochs is not None else {'simclr': 2000, 'barlowtwins': 10000}[args.ssl_method]
    if min(ssl_epochs, args.probe_epochs) < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error('Use positive epoch counts, a batch size of at least two, and nonnegative workers')
    if min(args.seed, *args.probe_seeds) < 0 or len(set(args.probe_seeds)) != len(args.probe_seeds):
        parser.error('Use nonnegative seeds and distinct probe seeds')
    if any(not 0 < labels <= 100 for labels in args.labels):
        parser.error('Label percentages must be in (0, 100]')

    # --- Read the checkpoint metadata (dataset and size are stored inside the distilled checkpoint) ------
    saved = {} if args.random_subset else torch.load(args.synthetic_checkpoint, map_location='cpu', weights_only=True)
    if isinstance(saved, torch.Tensor):  # a bare tensor of images, without metadata
        saved = {'data': saved, 'normalization': SPECS[args.dataset][1:] if args.dataset else None}
    if not isinstance(saved, dict) or not saved.get('complete', True):
        parser.error('Use a completed distillation checkpoint')
    source = saved.get('dataset', args.dataset)
    percentage = saved.get('percentage', args.percentage)
    if source not in SOURCES or not isinstance(percentage, (int, float)) or not 0 < percentage <= 100:
        parser.error('Supply a supported dataset and a percentage in (0, 100]')
    if args.dataset not in (None, source) or args.percentage not in (None, percentage):
        parser.error('The dataset and percentage must match the checkpoint metadata')

    # Resolve the evaluation targets (and fail early if a dataset folder is missing).
    targets = PAPER_TARGETS if 'all' in args.targets else [source if name == 'source' else name for name in args.targets]
    targets = list(dict.fromkeys(targets))  # drop duplicates, keep order
    for target in targets:
        locate(args.data_path, target)
    source_data = RawDataset(args.data_path, source, training=True)
    count = int(len(source_data) * percentage / 100)  # size of the pretraining set
    if count < 2:
        parser.error('The subset must contain at least two images')

    # Disable TF32 so results do not depend on the GPU generation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    size = INPUT_SIZE[source]

    # --- Build the pretraining image set -----------------------------------------------------------------
    if args.random_subset:
        # Baseline: a random subset of real training images, same size as the distilled set.
        indices = np.random.default_rng(args.seed + 1000).permutation(len(source_data))[:count].tolist()
        images = Subset(source_data, indices)
        image_source = dict(kind='random', indices=indices)
    else:
        pixels = saved.get('data')
        if not isinstance(pixels, torch.Tensor) or pixels.shape != (count, 3, size, size):
            parser.error(f'The checkpoint must contain {count} RGB images at {size}x{size}')
        if not pixels.is_floating_point() or not torch.isfinite(pixels).all():
            parser.error('The synthetic images must be finite floating-point values')
        normalization = saved.get('normalization') or SYNTHETIC_NORMALIZATION[source]
        mean, std = torch.as_tensor(normalization, dtype=pixels.dtype).reshape(2, 1, 3, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            parser.error('Invalid image normalization')
        # Undo the normalization: the SSL pipeline expects ordinary [0, 1] images.
        images = SyntheticImages((pixels.detach() * std + mean).clamp(0, 1).contiguous())
        image_source = dict(kind='synthetic', sha256=file_hash(args.synthetic_checkpoint))
        del saved, pixels

    # --- Pretrain, then probe every target at every label budget -----------------------------------------
    output_parent = Path('runs') / source / f'{percentage:g}' if args.random_subset else args.synthetic_checkpoint.parent
    mode = 'random' if args.random_subset else 'eval'
    output = args.output or output_parent / f'{mode}_{args.ssl_method}_{args.model}_seed{args.seed}'
    with exclusive_run(output):
        encoder, trained = train(images, source, args.model, args.ssl_method, args.seed, output,
                                 dict(source_fingerprint=source_data.fingerprint, image_source=image_source),
                                 epochs=ssl_epochs, batch_size=args.batch_size, workers=args.workers,
                                 device=args.device, checkpoint_every=100)
        summary = {}
        for target in targets:
            # When probing the source dataset itself, reuse the already loaded training split.
            train_set = source_data if target == source else RawDataset(args.data_path, target, training=True)
            test_set = RawDataset(args.data_path, target, training=False)
            summary[target] = {}
            for labels in dict.fromkeys(args.labels):
                scores = []
                for seed in args.probe_seeds:
                    result = linear_probe(encoder, trained['encoder_sha256'], source, target, train_set, test_set,
                                          labels, seed, output / 'probes' / f'{target}_lbl{labels:g}_seed{seed}.json',
                                          args.probe_epochs, args.workers, args.device)
                    scores.append(result['accuracy_percent'])
                mean = statistics.mean(scores)
                std = statistics.stdev(scores) if len(scores) > 1 else None
                summary[target][f'{labels:g}'] = dict(mean_percent=mean, std_percent=std, scores=scores)
                std_text = f' ± {std:.2f}' if std is not None else ''
                print(f'{target}, {labels:g}% labels: {mean:.2f}{std_text}%', flush=True)
        atomic_json(output / 'summary.json', dict(source=source, subset_percent=percentage,
                    ssl_method=args.ssl_method, ssl_seed=args.seed, probe_seeds=args.probe_seeds, results=summary))
    print(f'Results saved to {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
