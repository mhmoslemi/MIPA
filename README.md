# MIPA

Run these commands from the project root with Python 3.10+ and PyTorch, torchvision, NumPy, SciPy, and Pillow installed.

```sh
python -m MIPA.train_teacher --dataset CIFAR10
python -m MIPA.distill --dataset CIFAR10 --percentage 0.2
python -m MIPA.eval --synthetic-checkpoint runs/CIFAR10/0.2/synthetic.pt --targets all
```

Use `--data-path` to point to the data directory. The loaders read the extracted CIFAR Python archives, Tiny ImageNet in `tinyimagenet.pt` (`images_train`, `images_val`, `labels_train`, `labels_val`), and the official Aircraft and Stanford Dogs splits. For Tiny ImageNet, store images as RGB `uint8` tensors in NCHW order.

## Settings

- `train_teacher.py` trains a SimCLR ResNet-18 on the full source training split. Change `--epochs`, `--batch-size`, and `--seed` as needed.
- `distill.py` freezes the teacher and optimizes a randomly initialized subset of real images. Adjust `--iterations`, `--lr-img`, `--batch-size`, or `--temperature` directly. Use `--teacher FILE` for a different checkpoint or `--output` for a separate run.
- `eval.py` trains an encoder from the synthetic images and freezes it for linear probing. Use `--targets source` for same-dataset evaluation, `--targets all` for the paper's transfer targets, or list targets explicitly. Adjust `--ssl-epochs`, `--probe-epochs`, and `--labels` as needed. Results go in `summary.json`.

Defaults live in Python. Distillation defaults to 10,000, 8,000, and 6,000 iterations at 0.2%, 1%, and 2% respectively. Other sizes default to 10,000 iterations. Evaluation defaults to 2,000 SimCLR epochs and five probes per target and label budget (1% and 5%).
