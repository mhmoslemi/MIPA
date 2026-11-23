"""Distill a dataset by maximizing cross-view prototype mutual information."""

# MIPA (Mutual Information Prototype Alignment) in one paragraph:
#   * A frozen SimCLR teacher embeds both real images and the synthetic images (the "prototypes").
#   * For every real image, a softmax over cosine similarities gives its soft assignment to the prototypes.
#   * Two independently augmented views of the same real image should receive consistent assignments.
#     We therefore build the joint distribution P between the two views' assignments and maximize its
#     mutual information I(C1; C2) with respect to the synthetic *pixels* (the teacher is never updated).
#
# Outputs (under runs/<dataset>/<percentage>/ by default):
#   synthetic.pt         distilled images + metadata + full optimizer/RNG state, so a run can be resumed
#   prototypes_<it>.png  image grid saved after the first and the last iteration

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from torchvision.utils import save_image

# Allow both `python -m MIPA.distill` (package imports) and `python distill.py` (plain imports).
if __package__:
    from .common import atomic_save, exclusive_run, restore_rng, rng_state
    from .data import INPUT_SIZE, SOURCES, SYNTHETIC_NORMALIZATION, ImageView, RawDataset
    from .dd_teachers import load_teacher
    from .dm_dsa import make_diff_augmenter
    from .utils import AUGMENT_FNS, ParamDiffAug, get_time
else:
    from common import atomic_save, exclusive_run, restore_rng, rng_state
    from data import INPUT_SIZE, SOURCES, SYNTHETIC_NORMALIZATION, ImageView, RawDataset
    from dd_teachers import load_teacher
    from dm_dsa import make_diff_augmenter
    from utils import AUGMENT_FNS, ParamDiffAug, get_time


# Number of optimization steps used in the paper, keyed by synthetic-set size (% of the training split).
PAPER_ITERATIONS = {0.2: 10000, 1: 8000, 2: 6000}


def set_random_seed(seed):
    """Seed every RNG that distillation touches (Python, NumPy, PyTorch CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bipartite_view_transition_loss(features_u_real, features_v_real, features_u_syn, features_v_syn, tau, beta=1.1, pool_size=0):
    """Negative cross-view mutual information between prototype assignments.

    Args:
        features_u_real, features_v_real: (B, d) teacher features of the real batch under view u and view v.
        features_u_syn, features_v_syn:   (m, d) teacher features of the m synthetic images under the same two
                                          views (same augmentation draw as the real images of that view).
        tau:       assignment temperature of the softmax over cosine similarities.
        beta:      weight of the marginal-entropy terms. beta = 1 gives exactly -I(C1; C2).
        pool_size: if > 0, split the prototypes into balanced pools of about this size and average the loss
                   over pools (0 = a single pool containing every prototype).

    Returns:
        Scalar loss = H(P) - beta * (H(row marginal) + H(column marginal)); minimizing it maximizes the
        mutual information between the two views' prototype assignments.
    """
    # Cosine similarity = dot product of L2-normalized features.
    u_r = F.normalize(features_u_real, dim=1)
    v_r = F.normalize(features_v_real, dim=1)
    u_s = F.normalize(features_u_syn, dim=1)
    v_s = F.normalize(features_v_syn, dim=1)

    num_pools = math.ceil(len(u_s) / pool_size) if pool_size else 1
    losses = []
    # Fixed, balanced pools preserve image order and include every prototype.
    for us, vs in zip(u_s.tensor_split(num_pools), v_s.tensor_split(num_pools)):
        sim_u = torch.mm(u_r, us.t())  # (B, pool): real image vs. prototype, view u
        sim_v = torch.mm(v_r, vs.t())  # (B, pool): real image vs. prototype, view v
        # Keep the effective sharpness comparable across set sizes: tau is used as-is for 100 prototypes
        # and shrinks like 1 / log(m) for larger sets (and for pools).
        effective_tau = tau * math.log(100.0) / math.log(max(len(us), 100))
        p_u = F.softmax(sim_u / effective_tau, dim=1)  # soft assignment of each real image, view u
        p_v = F.softmax(sim_v / effective_tau, dim=1)  # soft assignment of each real image, view v

        # Joint distribution over (prototype in view u, prototype in view v), averaged over the batch,
        # then symmetrized so both view orders are treated equally (equal row/column marginals).
        P = torch.mm(p_u.t(), p_v) / len(p_u)
        P = 0.5 * (P + P.t())
        P_i = P.sum(dim=1, keepdim=True)  # marginal over view-u prototypes
        P_j = P.sum(dim=0, keepdim=True)  # marginal over view-v prototypes
        tiny = torch.finfo(P.dtype).tiny  # clamp so log(0) never produces NaN
        joint_entropy = -(P * P.clamp_min(tiny).log()).sum()
        marginal_entropy = -(P_i * P_i.clamp_min(tiny).log()).sum() - (P_j * P_j.clamp_min(tiny).log()).sum()
        # I(C1; C2) = H(C1) + H(C2) - H(C1, C2), so this is -I when beta == 1.
        losses.append(joint_entropy - beta * marginal_entropy)
    return torch.stack(losses).mean()


def distill(args, output):
    """Run (or resume) one distillation job and write its checkpoint to `output`."""
    set_random_seed(args.seed)

    # --- Real data and the frozen teacher --------------------------------------------------------------
    raw = RawDataset(args.data_path, args.dataset, training=True)
    normalization = SYNTHETIC_NORMALIZATION[args.dataset]
    real_data = ImageView(raw, transforms.Compose([transforms.ToTensor(), transforms.Normalize(*normalization)]))
    count = int(len(real_data) * args.percentage / 100)  # number of synthetic images
    if count < 2 or not 1 <= args.batch_real <= len(real_data):
        raise ValueError('Use at least two synthetic images and a real batch that fits the training split')
    size = INPUT_SIZE[args.dataset]
    teacher, teacher_seed, teacher_hash = load_teacher(args.teacher, args.dataset, normalization, size, args.device)
    print(f'Loaded frozen teacher seed {teacher_seed}', flush=True)

    # The settings that define a run; a checkpoint is only resumed if all of them match exactly.
    run_args = {key: getattr(args, key) for key in (
        'dataset', 'percentage', 'seed', 'Iteration', 'lr_img', 'batch_real', 'temperature', 'beta',
        'prototype_pool_size', 'distill_aug_strategy', 'feature_batch')}
    run_args['single_channel'] = False

    # --- Resume handling -------------------------------------------------------------------------------
    result_path = output / 'synthetic.pt'
    legacy_path = output / f'res_IIC_{args.dataset}_ResNet18BN_{args.percentage:g}percent_final.pt'
    if not result_path.exists() and legacy_path.exists():
        result_path = legacy_path  # accept checkpoints written under the old file name
    saved = None
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f'{result_path} exists. Use --resume or choose a new output folder.')
        saved = torch.load(result_path, map_location='cpu', weights_only=True)
        saved_args = {key: value for key, value in saved['run_args'].items() if key in run_args}
        saved_args.setdefault('prototype_pool_size', 0)  # older checkpoints predate this option
        # Refuse to continue from a checkpoint made with a different recipe, teacher, data or shape.
        if (saved_args != run_args or saved.get('init') != 'random' or saved.get('teacher_seed') != teacher_seed
                or saved.get('teacher_sha256', teacher_hash) != teacher_hash
                or saved.get('source_fingerprint', raw.fingerprint) != raw.fingerprint
                or saved['normalization'] != normalization or saved['data'].shape != (count, 3, size, size)
                or not 0 <= saved['iteration'] <= args.Iteration):
            raise ValueError('The checkpoint does not match this dataset, teacher, or distillation recipe')
        if saved['complete']:
            print(f'Completed checkpoint: {result_path}', flush=True)
            return

    # --- Synthetic images: initialize from random real images (or restore from the checkpoint) -----------
    if saved is None:
        indices = np.random.default_rng(args.seed + 1000).permutation(len(real_data))[:count]
        initial = torch.stack([real_data[int(index)][0] for index in indices]).to(args.device)
        images = initial.clone().requires_grad_(True)  # these pixels are the only learnable parameters
    else:
        initial = saved['initial_data'].to(args.device)
        images = saved['data'].to(args.device).detach().requires_grad_(True)

    # Valid pixel range expressed in normalized space, used to clamp the images after every step.
    mean, std = (torch.tensor(value, device=args.device).view(1, 3, 1, 1) for value in normalization)
    pixel_min, pixel_max = -mean / std, (1 - mean) / std
    # 3x3 Gaussian kernel (applied per channel) used to smooth the pixel gradients.
    pixel_filter = images.new_tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16
    pixel_filter = pixel_filter.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    augmenter = make_diff_augmenter(args.distill_aug_strategy, ParamDiffAug())

    def embed(batch):
        """Teacher features in chunks of --feature-batch; images that need gradients use checkpointing
        so activations are recomputed in the backward pass instead of being stored (saves memory)."""
        return torch.cat([checkpoint(teacher.embed, block, use_reentrant=False) if block.requires_grad
                          else teacher.embed(block) for block in batch.split(args.feature_batch)])

    # --- Real-data ordering, optimizer and learning-rate schedule ---------------------------------------
    data_rng = torch.Generator().manual_seed(args.seed + 20000)
    real_order = torch.randperm(len(real_data), generator=data_rng)  # one pass over the data per "epoch"
    real_position = 0
    objective_rng = np.random.default_rng(args.seed + 10000)  # draws the augmentation seeds
    optimizer = torch.optim.Adam([images], lr=args.lr_img, eps=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Iteration, eta_min=args.lr_img * 0.01)
    start = 0
    history = []  # loss value of every iteration so far
    if saved is not None:
        # Restore the exact optimizer, schedule, RNG streams and data position, so the resumed run
        # continues as if it had never been interrupted.
        start = saved['iteration']
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        objective_rng.bit_generator.state = saved['objective_rng']
        data_rng.set_state(saved['data_rng'])
        real_order, real_position = saved['real_order'], saved['real_position']
        restore_rng(saved['rng'])
        numpy_rng = saved['numpy_rng']
        np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
        history = saved['loss_history'].tolist()
        if len(history) != start or not 0 <= real_position <= len(real_order) == len(real_data):
            raise ValueError('Invalid distillation checkpoint state')
        del saved

    def real_batches():
        """Iterate over the not-yet-used part of the current data order in full batches."""
        return iter(torch.utils.data.DataLoader(
            real_data, batch_size=args.batch_real, sampler=real_order[real_position:].tolist(),
            num_workers=0, pin_memory=torch.device(args.device).type == 'cuda', drop_last=True,
            generator=torch.Generator().manual_seed(args.seed)))

    def save(iteration):
        """Write the resumable checkpoint (atomically) after `iteration` steps."""
        numpy_rng = np.random.get_state()
        atomic_save(result_path, {
            'data': images.detach().cpu(), 'initial_data': initial.detach().cpu(),
            'dataset': args.dataset, 'percentage': args.percentage, 'normalization': normalization,
            'method': 'Bipartite_ViewTransition_MI', 'iteration': iteration, 'complete': iteration == args.Iteration,
            'run_args': run_args, 'init': 'random', 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'rng': rng_state(), 'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            'objective_rng': objective_rng.bit_generator.state, 'data_rng': data_rng.get_state(),
            'real_order': real_order, 'real_position': real_position,
            'teacher_source': 'pretrained', 'teacher_seed': teacher_seed, 'teacher_sha256': teacher_hash,
            'source_fingerprint': raw.fingerprint, 'seed': args.seed, 'model': 'ResNet18BN',
            'loss_history': torch.tensor(history, dtype=torch.float64),
        })

    samples = real_batches()
    if not result_path.exists():
        save(0)
    print(f'{args.dataset}, {count} synthetic images, {args.Iteration} iterations, batch {args.batch_real}, '
          f'learning rate {args.lr_img:g}, temperature {args.temperature:g}', flush=True)

    # --- Main optimization loop ------------------------------------------------------------------------
    for iteration in range(start + 1, args.Iteration + 1):
        # Next real batch; start a fresh random pass over the data when the current one is exhausted.
        try:
            real, _ = next(samples)
        except StopIteration:
            real_order = torch.randperm(len(real_data), generator=data_rng)
            real_position = 0
            samples = real_batches()
            real, _ = next(samples)
        real = real.to(args.device)
        real_position += len(real)

        optimizer.zero_grad(set_to_none=True)
        # One augmentation seed per view. Real and synthetic images of the same view share the seed, so
        # they are compared under identical transformations (one transformation per view and batch).
        seed_u, seed_v = (int(objective_rng.integers(0, 2**31 - 1)) for _ in range(2))
        with torch.no_grad():  # the real side is a fixed target: no gradients needed
            real_u = embed(augmenter(real, seed_u))
            real_v = embed(augmenter(real, seed_v))
        # Gradients flow from the loss through the (differentiable) augmentation into the synthetic pixels.
        syn_u = embed(augmenter(images, seed_u))
        syn_v = embed(augmenter(images, seed_v))
        loss = bipartite_view_transition_loss(real_u, real_v, syn_u, syn_v,
                                              args.temperature, args.beta, args.prototype_pool_size)
        loss.backward()
        # Smooth the pixel gradients before applying the paper's Adam update.
        images.grad.copy_(F.conv2d(images.grad, pixel_filter, padding=1, groups=3))
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            images.copy_(torch.maximum(torch.minimum(images, pixel_max), pixel_min))  # keep pixels valid
        history.append(loss.item())
        del loss, real_u, real_v, syn_u, syn_v  # free the graph before the next iteration
        if iteration % 100 == 0:
            print(f'{get_time()} Iteration {iteration}: loss {history[-1]:.6f}', flush=True)
        if iteration in (1, args.Iteration):
            # Undo the normalization to save a viewable PNG grid of the current prototypes.
            visible = (images.detach() * std + mean).clamp(0, 1).cpu()
            save_image(visible, output / f'prototypes_{iteration}.png', nrow=math.ceil(math.sqrt(count)))
        if iteration == 1 or iteration % args.save_every == 0 or iteration == args.Iteration:
            save(iteration)
    print(f'Synthetic images saved to {result_path}', flush=True)


def main(argv=None):
    """Parse the command line, fill in the paper's defaults, and run `distill` under an exclusive lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=SOURCES, required=True)
    parser.add_argument('--percentage', type=float, default=0.2)  # synthetic-set size, % of the training split
    parser.add_argument('--data-path', '--data_path', default='data')
    parser.add_argument('--teacher', type=Path, help='Default: runs/<dataset>/teacher/encoder.pt')
    parser.add_argument('--output', '--save_path', type=Path, help='Default: runs/<dataset>/<percentage>')
    parser.add_argument('--iterations', '--Iteration', dest='Iteration', type=int)
    parser.add_argument('--lr-img', '--lr_img', dest='lr_img', type=float)
    parser.add_argument('--batch-size', '--batch_real', dest='batch_real', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.2)  # assignment temperature tau
    parser.add_argument('--beta', type=float, default=1.0)  # 1.0 = exact mutual information
    parser.add_argument('--prototype-pool-size', '--prototype_pool_size', type=int, default=0)
    parser.add_argument('--augmentation', '--distill_aug_strategy', dest='distill_aug_strategy',
                        default='color_crop_cutout_scale')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--feature-batch', '--feature_batch', type=int, default=128)  # chunk size for teacher forward passes
    parser.add_argument('--save-every', '--save_every', type=int, default=500)  # checkpoint interval in iterations
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.percentage <= 100:
        parser.error('The percentage must be in (0, 100]')
    # Defaults that depend on other arguments.
    if args.Iteration is None:
        args.Iteration = PAPER_ITERATIONS.get(args.percentage, 10000)
    if args.lr_img is None:
        args.lr_img = 0.001 if args.dataset == 'TinyImageNet' and args.percentage == 2 else 0.0015
    if min(args.Iteration, args.batch_real, args.save_every, args.feature_batch) < 1 or args.seed < 0:
        parser.error('Use positive iteration and batch counts and a nonnegative seed')
    if any(not math.isfinite(value) or value <= 0 for value in (args.lr_img, args.temperature, args.beta)):
        parser.error('The learning rate, temperature, and beta must be finite and positive')
    if args.prototype_pool_size == 1 or args.prototype_pool_size < 0:
        parser.error('Prototype pool size must be zero or at least two')
    if args.distill_aug_strategy.lower() != 'none' and set(args.distill_aug_strategy.split('_')) - AUGMENT_FNS.keys():
        parser.error('Unknown augmentation operation')
    runs = Path('runs') / args.dataset
    args.teacher = args.teacher or runs / 'teacher' / 'encoder.pt'
    output = args.output or runs / f'{args.percentage:g}'
    with exclusive_run(output):
        distill(args, output)


if __name__ == '__main__':
    main()
