"""Dataset readers, augmentation views, and label sampling for evaluation."""

# Everything related to data lives here:
#   * locating and reading the five datasets (CIFAR-10/100, Tiny ImageNet, FGVC Aircraft, Stanford Dogs)
#     with their official train/test splits;
#   * the SSL augmentation pipelines (SimCLR / Barlow Twins) and the linear-probe transforms;
#   * class-balanced selection of the labeled subset used by the linear probe.

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

from utils.common import digest_json, file_hash, worker_seed


# Datasets that can be distilled, and the full list of evaluation (transfer) targets used in the paper.
SOURCES = ('CIFAR10', 'CIFAR100', 'TinyImageNet')
PAPER_TARGETS = (*SOURCES, 'Aircraft', 'Dogs')
# Native image resolution of each source dataset.
INPUT_SIZE = {'CIFAR10': 32, 'CIFAR100': 32, 'TinyImageNet': 64}
# [mean, std] per channel used for SSL training and probing.
SSL_NORMALIZATION = {
    'CIFAR10': [[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]],
    'CIFAR100': [[0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]],
    'TinyImageNet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}
# [mean, std] in which the synthetic images are stored (and optimized) during distillation.
SYNTHETIC_NORMALIZATION = {
    'CIFAR10': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    'CIFAR100': [[0.5071, 0.4866, 0.4409], [0.2673, 0.2564, 0.2762]],
    'TinyImageNet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}


# name -> (number of classes, channel mean, channel std)
SPECS = {
    'CIFAR10': (10, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    'CIFAR100': (100, [0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023]),
    'Aircraft': (100, [0.4804, 0.5116, 0.5349], [0.2021, 0.1953, 0.2297]),
    'CUB2011': (200, [0.4857, 0.4995, 0.4324], [0.2145, 0.2098, 0.2496]),
    'Dogs': (120, [0.4765, 0.4516, 0.3911], [0.2490, 0.2435, 0.2479]),
    'Flowers': (102, [0.4329, 0.3820, 0.2965], [0.2828, 0.2333, 0.2615]),
    'TinyImageNet': (200, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def locate(root, dataset):
    """Find the folder that holds `dataset` under `root` by checking for a known marker file.

    Several layouts are accepted per dataset; exactly one must exist, otherwise the choice would be
    ambiguous. Raises FileNotFoundError (with the expected path) if the dataset is not staged.
    """
    root = Path(root)
    # (folder, marker file inside it) candidates for each dataset
    candidates = {
        'CIFAR10': [(root, 'cifar-10-batches-py/data_batch_1')],
        'CIFAR100': [(root, 'cifar-100-python/train')],
        'Aircraft': [(root / 'fgvc-aircraft-2013b/data', 'variants.txt'),
                     (root / 'aircraft/fgvc-aircraft-2013b/data', 'variants.txt')],
        'CUB2011': [(root / 'CUB_200_2011', 'images.txt'),
                    (root / 'cub2011/CUB_200_2011', 'images.txt')],
        'Dogs': [(root / 'StanfordDogs', 'train_list.mat'), (root / 'dogs', 'train_list.mat'), (root / 'stanford_dogs', 'train_list.mat'), (root, 'train_list.mat')],
        'Flowers': [(root / 'Oxford102Flowers', 'setid.mat'), (root / 'flowers-102', 'setid.mat'), (root / 'flowers/flowers-102', 'setid.mat')],
        'TinyImageNet': [(root, 'tinyimagenet.pt')],
    }[dataset]
    matches = [folder for folder, marker in candidates if (folder / marker).is_file()]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous {dataset} roots: {matches}; keep one canonical copy.')
    if not matches:
        raise FileNotFoundError(f'{dataset} is not staged. Expected {candidates[0][0] / candidates[0][1]}')
    return matches[0]


def read_pairs(path):
    """Read a text file of 'ID value' lines into {int ID: value string}."""
    return {int(key): value for key, value in (line.strip().split(maxsplit=1) for line in Path(path).read_text().splitlines() if line.strip())}


def records(root, dataset, training):
    """Read official split membership; never use test data for selection/training.

    Returns a list of (image path, zero-based class label) for the file-based datasets
    (Aircraft, CUB-2011, Dogs, Flowers). `training` selects the train or the test split.
    """
    folder = locate(root, dataset)
    if dataset == 'Aircraft':
        classes = (folder / 'variants.txt').read_text().splitlines()
        mapping = {name.strip(): index for index, name in enumerate(classes)}
        split = 'trainval' if training else 'test'  # the official train+val split is used for training
        samples = []
        for line in (folder / f'images_variant_{split}.txt').read_text().splitlines():
            name, label = line.split(maxsplit=1)
            samples.append((folder / 'images' / f'{name}.jpg', mapping[label.strip()]))
        return samples
    if dataset == 'CUB2011':
        names = read_pairs(folder / 'images.txt')
        labels = read_pairs(folder / 'image_class_labels.txt')
        splits = read_pairs(folder / 'train_test_split.txt')
        if names.keys() != labels.keys() or names.keys() != splits.keys():
            raise ValueError('CUB metadata image IDs disagree')
        return [(folder / 'images' / name, int(labels[index]) - 1)
                for index, name in names.items() if int(splits[index]) == int(training)]
    if dataset == 'Dogs':
        image_folder = folder / ('images' if (folder / 'images').is_dir() else 'Images')
        metadata = loadmat(folder / ('train_list.mat' if training else 'test_list.mat'))
        names = metadata.get('file_list', metadata.get('annotation_list'))
        if names is None:
            raise ValueError('Dogs split has neither file_list nor annotation_list')
        labels = metadata['labels'].reshape(-1)
        if len(names) != len(labels):
            raise ValueError('Dogs split image/label counts disagree')
        samples = []
        for entry, label in zip(names.reshape(-1), labels):
            # .mat files nest the file names in object arrays of varying depth; unwrap to a plain string
            while isinstance(entry, np.ndarray):
                entry = entry.item() if entry.size == 1 else ''.join(entry.tolist())
            name = str(entry)
            if not name.lower().endswith('.jpg'):
                name += '.jpg'
            samples.append((image_folder / name, int(label) - 1))
        return samples
    if dataset == 'Flowers':
        image_folder = folder / 'jpg' if (folder / 'jpg').is_dir() else folder
        # Follow existing repository protocol: official train only, not train+val.
        splits = loadmat(folder / 'setid.mat', squeeze_me=True)
        ids = np.atleast_1d(splits['trnid' if training else 'tstid']).astype(int)
        labels = np.atleast_1d(loadmat(folder / 'imagelabels.mat', squeeze_me=True)['labels'])
        return [(image_folder / f'image_{index:05d}.jpg', int(labels[index - 1]) - 1) for index in ids]
    raise ValueError(f'{dataset} does not use image-file records')


def select_labels(labels, percentage, seed, policy='exact'):
    """Select a nested label subset with an exact overall budget.

    Classes are visited in a random (seeded) order and, round after round, one image per class is added
    until the budget is reached, which keeps the subset as class-balanced as possible. A larger budget
    with the same seed always contains the smaller one ("nested").

    Policies: 'exact' (default) respects the budget exactly, 'at_least_one_per_class' raises it to the
    number of classes if needed, 'random' ignores class balance.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    count = max(1, int(len(labels) * percentage / 100))
    generator = np.random.default_rng(seed + 5000)
    if policy == 'random':
        return generator.permutation(len(labels))[:count].astype(np.int64, copy=False)
    if policy == 'at_least_one_per_class':
        count = max(count, len(classes))
    elif policy != 'exact':
        raise ValueError(policy)
    if count > len(labels):
        raise ValueError('Label budget exceeds the target training split')
    class_order = generator.permutation(classes)
    pools = [generator.permutation(np.flatnonzero(labels == label)) for label in class_order]  # shuffled images per class
    chosen = []
    level = 0  # round-robin depth: take the `level`-th image of each class in turn
    while len(chosen) < count:
        for pool in pools:
            if level < len(pool):
                chosen.append(int(pool[level]))
                if len(chosen) == count:
                    break
        level += 1
    return np.asarray(chosen, dtype=np.int64)


class RawDataset(Dataset):
    """A dataset split that returns (PIL image, label) and carries a content fingerprint.

    The fingerprint (hash of the labels plus an image hash or file metadata) is stored in checkpoints so
    that results can be tied to the exact data they were produced from.
    """

    def __init__(self, root, name, training):
        self.name = name
        self.classes = SPECS[name][0]
        self.images = self.samples = self.base = None
        folder = locate(root, name)
        if name in ('CIFAR10', 'CIFAR100'):
            cls = getattr(datasets, name)
            self.base = cls(str(folder), train=training, download=False)  # data must already be on disk
            self.targets = np.asarray(self.base.targets, dtype=np.int64)
            pixels = hashlib.sha256(self.base.data.tobytes()).hexdigest()
            identity = {'pixels_sha256': pixels}
        elif name == 'TinyImageNet':
            path = folder / 'tinyimagenet.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)  # memory-mapped, not fully loaded
            split = 'train' if training else 'val'
            self.images = saved['images_' + split]
            self.targets = saved['labels_' + split].long().numpy()
            if self.images.dtype != torch.uint8 or tuple(self.images.shape[1:]) != (3, 64, 64):
                raise ValueError('Expected native 64x64 uint8 Tiny ImageNet archive')
            identity = {'archive_sha256': file_hash(path), 'split': split}
        else:
            self.samples = records(root, name, training)
            self.targets = np.asarray([label for _, label in self.samples], dtype=np.int64)
            # File metadata fingerprint: explicitly NOT claimed as a pixel hash.
            identity = {'files_size_mtime': [
                (str(path.relative_to(folder)), label, path.stat().st_size, path.stat().st_mtime_ns)
                for path, label in self.samples]}
        if len(self.targets) == 0 or self.targets.min() < 0 or self.targets.max() >= self.classes:
            raise ValueError(f'Invalid labels in {name}')
        if self.images is not None and len(self.images) != len(self.targets):
            raise ValueError('Image/label count mismatch')
        self.fingerprint = digest_json({'name': name, 'training': training,
                                       'labels': self.targets.tolist(), **identity})

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.base is not None:
            image = self.base[index][0]  # CIFAR: already a PIL image
        elif self.images is not None:
            image = T.functional.to_pil_image(self.images[index])  # Tiny ImageNet: uint8 tensor -> PIL
        else:
            with Image.open(self.samples[index][0]) as handle:
                image = handle.convert('RGB')
        return image, int(self.targets[index])


class ImageView(Dataset):
    """Apply a transform to the image of every (image, label) pair of `base`."""

    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return self.transform(image), label


class GaussianBlur:
    """SimCLR-style blur: Gaussian blur with a random sigma drawn uniformly from `sigma` = (low, high)."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(random.uniform(*self.sigma)))


class TwoViews:
    """Apply two different transform pipelines to the same image and return both results."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return tuple(transform(image) for transform in self.transforms)


class ProbeCrop:
    """Keep native CIFAR/Tiny test images intact; crop larger transfer images."""
    def __init__(self, size, ratio):
        self.size = size
        self.transform = T.Compose([T.Resize(round(size * ratio), interpolation=T.InterpolationMode.BICUBIC),
                                    T.CenterCrop(size)])

    def __call__(self, image):
        return image if image.size == (self.size, self.size) else self.transform(image)


def ssl_transform(source, method):
    """Two-view SSL augmentation for `source`, tuned per method.

    SimCLR: strong color jitter and Gaussian blur (p = 0.5) on both views. Barlow Twins: milder color
    jitter, no blur, and solarization (p = 0.2) on the second view only.
    Returns a callable that maps one PIL image to a tuple of two augmented tensors.
    """
    if method == 'simclr':
        jitter, blur, solarize = (0.8, 0.8, 0.8, 0.2), (0.5, 0.5), (0.0, 0.0)
    else:
        jitter, blur, solarize = (0.4, 0.4, 0.2, 0.1), (0.0, 0.0), (0.0, 0.2)
    views = []
    for i in range(2):  # blur and solarize probabilities can differ between the two views
        transforms = [
            T.RandomResizedCrop(INPUT_SIZE[source], interpolation=T.InterpolationMode.BICUBIC),
            T.RandomApply([T.ColorJitter(*jitter)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur((0.1, 2.0))], p=blur[i]),
            T.RandomSolarize(128, p=solarize[i]),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(), T.Normalize(*SSL_NORMALIZATION[source])]
        views.append(T.Compose(transforms))
    return TwoViews(views)


def probe_transform(source, training):
    """Transform for the linear probe: light augmentation for training, center crop for testing."""
    size = INPUT_SIZE[source]
    if training:
        operations = [T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip()]
    else:
        operations = [ProbeCrop(size, 8 / 7)]  # resize to 8/7 of the size, then center-crop
    return T.Compose(operations + [T.ToTensor(), T.Normalize(*SSL_NORMALIZATION[source])])


def loader(dataset, batch, workers, seed, shuffle=False, drop_last=False):
    """DataLoader whose shuffling and worker randomness are fully determined by `seed`."""
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=drop_last,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed))


def label_indices(dataset, percentage, seed):
    """Indices of the labeled training subset that the linear probe may use."""
    # Reuse the exact, nested, class-balanced repository selector. It never
    # expands the requested global budget to manufacture one label per class.
    return select_labels(dataset.targets, percentage, seed, policy='exact').tolist()
