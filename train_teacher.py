"""Train a ResNet-18 SimCLR teacher on the full training split."""

# The teacher is the frozen "measuring stick" used by distill.py: real and synthetic images are both
# embedded with it. It is trained once per dataset on the full, unlabeled training split.
# Output: runs/<dataset>/teacher/encoder.pt (plus a resumable training.pt and a per-epoch ssl.jsonl log).

import argparse
from pathlib import Path

import torch

# Allow both `python -m MIPA.train_teacher` (package imports) and `python train_teacher.py` (plain imports).
if __package__:
    from .common import TEACHER_SEEDS, exclusive_run
    from .data import SOURCES, RawDataset
    from .ssl_training import train
else:
    from common import TEACHER_SEEDS, exclusive_run
    from data import SOURCES, RawDataset
    from ssl_training import train


def main(argv=None):
    """Parse arguments and pretrain the teacher with SimCLR (resumes automatically if interrupted)."""
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

    # Disable TF32 so results do not depend on the GPU generation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = RawDataset(args.data_path, args.dataset, training=True)  # labels are never used for SSL
    output = args.output or Path('runs') / args.dataset / 'teacher'
    with exclusive_run(output):
        checkpoint, _ = train(dataset, args.dataset, 'ResNet18', 'simclr', seed,
                              output, dict(source_fingerprint=dataset.fingerprint), epochs=args.epochs,
                              batch_size=args.batch_size, workers=args.workers, device=args.device,
                              checkpoint_every=50)  # save a resumable checkpoint every 50 epochs
    print(f'Teacher saved to {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
