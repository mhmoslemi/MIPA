"""Train a ResNet-18 SimCLR teacher on the full training split."""

import argparse
from pathlib import Path

import torch

if __package__:
    from .common import TEACHER_SEEDS, exclusive_run
    from .data import SOURCES, RawDataset
    from .ssl_training import train
else:
    from common import TEACHER_SEEDS, exclusive_run
    from data import SOURCES, RawDataset
    from ssl_training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=SOURCES, required=True)
    parser.add_argument('--data-path', default='data')
    parser.add_argument('--output', type=Path, help='Default: runs/<dataset>/teacher')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, help='Defaults to the seed used in the paper for this dataset')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error('Use at least one epoch, a batch size of at least two, and nonnegative workers')
    seed = TEACHER_SEEDS[args.dataset] if args.seed is None else args.seed
    if seed < 0:
        parser.error('The seed must be nonnegative')

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = RawDataset(args.data_path, args.dataset, training=True)
    output = args.output or Path('runs') / args.dataset / 'teacher'
    with exclusive_run(output):
        checkpoint, _ = train(dataset, args.dataset, 'ResNet18', 'simclr', seed,
                              output, dict(source_fingerprint=dataset.fingerprint), epochs=args.epochs,
                              batch_size=args.batch_size, workers=args.workers, device=args.device,
                              checkpoint_every=50)
    print(f'Teacher saved to {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
