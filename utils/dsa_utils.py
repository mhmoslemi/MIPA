"""Helper utilities from the dataset-distillation benchmark code, plus the DSA augmentations that MIPA uses.

Used by MIPA (distill.py and dm_dsa.py import these):
    * ``ParamDiffAug``  - parameter container for the differentiable augmentation,
    * ``DiffAugment``   - applies a differentiable (optionally Siamese) augmentation strategy to a batch,
    * ``rand_scale``, ``rand_rotate``, ``rand_flip``, ``rand_brightness``, ``rand_saturation``,
      ``rand_contrast``, ``rand_crop``, ``rand_cutout`` and ``set_seed_DiffAug`` - the pieces
      ``DiffAugment`` calls,
    * ``AUGMENT_FNS``   - maps strategy names ('color', 'crop', 'cutout', 'flip', 'scale', 'rotate') to
      those functions (distill.py also uses its keys to validate the augmentation strategy),
    * ``get_time``      - timestamp string for log lines.

Everything else is legacy code from the original gradient-matching / distribution-matching benchmarks
and is not used by MIPA: dataset loading (get_dataset, TensorDataset), the model factory (get_network,
get_default_convnet_setting, get_eval_pool), gradient matching (distance_wb, match_loss, get_loops), the
non-differentiable DC augmentation (augment, get_daparam), evaluation of synthetic sets (epoch,
evaluate_synset, evaluate_synset_SSL, init_ssl, epoch_ssl) and the optimal-transport losses
(get_transport_plan, transport_*_loss).

At import time this module loads the older backbones from networks.py (not benchmark_networks.py), so
networks.py is imported whenever MIPA imports utils, although none of its classes are used.

Image convention: batches are float tensors of shape (N, C, H, W). In MIPA, DiffAugment receives images in
dataset-normalized space (see dm_dsa.DMDSA).
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage import rotate as scipyrotate
from models.networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN

def get_dataset(dataset, data_path):
    """Load a dataset by name and return its metadata, train/test sets and a test loader.

    ``dataset`` is one of 'MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'TinyImageNet' (the last one
    is read from ``data_path``/tinyimagenet.pt). Images are standardized per channel with ``mean``/``std`` and
    the training set has no augmentation.

    Returns:
        (channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader)

    Not used by MIPA.
    """
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        # ToTensor scales pixels to [0, 1]; Normalize then standardizes each channel with the dataset mean/std
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        # the usual ImageNet channel statistics
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Tiny ImageNet comes from a preprocessed tensor file of raw [0, 255] images, shape (N, 3, 64, 64)
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        # raw [0, 255] pixels -> float [0, 1]; the per-channel standardization follows below
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit('unknown dataset: %s'%dataset)


    # fixed-order test loader, reused by every evaluation
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader



class TensorDataset(Dataset):
    """In-memory dataset of (image, label) pairs backed by two tensors. Not used by MIPA."""
    def __init__(self, images, labels): # images: n x c x h x w tensor
        """Store detached copies of ``images`` (float, N x C x H x W) and ``labels`` (N,)."""
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        """Return the (image, label) pair at ``index``."""
        return self.images[index], self.labels[index]

    def __len__(self):
        """Return the number of samples."""
        return self.images.shape[0]



def get_default_convnet_setting():
    """Return the default ConvNet setting (width 128, depth 3, relu, instancenorm, avgpooling). Not used by MIPA."""
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), seed=None):
    """Build a benchmark network by name, seed its initialization and move it to the GPU(s) if available.

    ``model`` is a name such as 'ConvNet', 'ResNet18', 'VGG11BN' or a ConvNet ablation variant (see the
    section comments in the body); unknown names exit. ``im_size`` only matters for the ConvNet variants. The
    result is wrapped in nn.DataParallel when more than one GPU is visible. MIPA builds its encoders in
    common.build_encoder instead. Not used by MIPA.
    """
    if seed is None:
        # time-based seed in [0, 99999] so that every call gives a different initialization
        seed = int(time.time() * 1000) % 100000
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # default ConvNet settings; each ConvNet<X> name below overrides exactly one of them
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    # depth ablation: ConvNetD<k> has k conv blocks
    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    # width ablation: ConvNetW<k> has k channels per conv layer
    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    # activation ablation: AS = sigmoid, AR = relu, AL = leakyrelu, ASwish = swish (ASwishBN also uses batchnorm)
    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    # normalization ablation: NN = none, BN = batchnorm, LN = layernorm, IN = instancenorm, GN = groupnorm
    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    # pooling ablation: NP = no pooling, MP = max pooling, AP = average pooling
    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    else:
        net = None
        exit('unknown model: %s'%model)

    # use CUDA when available; with several GPUs the net is wrapped in DataParallel (which is why the SSL helpers
    # below reach the backbone through ``net.module``)
    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def get_time():
    """Return the current local time as a '[YYYY-MM-DD HH:MM:SS]' string, used as a prefix for log lines.

    Used by MIPA (distill.py).
    """
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    """Layer-wise cosine distance between real and synthetic gradients ``gwr``, ``gws`` of one parameter tensor.

    Each output unit (row after reshaping to (out, rest)) contributes 1 - cos(gwr_k, gws_k); the rows are
    summed. 1-D parameters (biases, norm scale/shift) are ignored and give 0. Not used by MIPA.
    """
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        # 2-D (linear) weights are already (out, in): nothing to reshape
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        # 1-D parameters are not matched: a zero distance is returned below (so these two reshapes have no effect)
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    # per output unit k: 1 - <gwr_k, gws_k> / (|gwr_k| * |gws_k| + 1e-6), summed over k
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    """Distance between two lists of per-layer gradients (``gw_syn``, ``gw_real``) according to ``args.dis_metric``.

    'ours' sums distance_wb over layers, 'mse' is the squared L2 distance of the concatenated gradients and
    'cos' is the cosine distance of the concatenated gradients. Not used by MIPA.
    """
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        # flatten every layer's gradient and concatenate them into one long vector
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        # flatten every layer's gradient and concatenate them into one long vector
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        # cosine distance 1 - cos(real, syn) of the two gradient vectors (1e-6 avoids division by zero)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis



def get_loops(ipc):
    """Return (outer_loop, inner_loop) for gradient matching given images per class ``ipc``. Not used by MIPA."""
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    """Run one training ('train') or evaluation pass over ``dataloader`` and return (mean loss, accuracy).

    With ``aug=True`` the images are augmented by DiffAugment (if ``args.dsa``) or by the DC ``augment``
    function. Not used by MIPA.
    """
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        # number of correct predictions in this batch
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        # weight by batch size so that the final division gives a per-sample average
        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    """Train ``net`` from scratch on a synthetic set and return (net, train accuracy, test accuracy on real data).

    Uses SGD (momentum 0.9, weight decay 5e-4) with learning rate ``args.lr_net`` for ``args.epoch_eval_train``
    epochs, decayed 10x halfway; images are augmented. ``images_train`` is (N, C, H, W), ``labels_train`` (N,).
    Not used by MIPA.
    """
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    # the learning rate is dropped 10x once, after epoch Epoch//2 + 1
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
        if ep in lr_schedule:
            # decay the learning rate and re-create the optimizer (this also resets the momentum buffers)
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test



def init_ssl(net, images_train, args):
    """Create the SSL projection head and an SGD optimizer over backbone and head; return (projector, optimizer).

    The backbone must provide ``embed(x)``. Not used by MIPA.
    """
    # DataParallel hides the real network behind ``.module``
    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed
    with torch.no_grad():
        # probe one image to find the size of the backbone's embedding
        feature_dim = embed(images_train[:1]).shape[1]

    # two-layer MLP projection head: feature_dim -> projection_dim -> projection_dim (SimCLR style)
    projector = nn.Sequential(
        nn.Linear(feature_dim, args.projection_dim),
        nn.ReLU(inplace=True),
        nn.Linear(args.projection_dim, args.projection_dim),
    ).to(args.device)

    parameters = list(net.parameters()) + list(projector.parameters())
    optimizer = torch.optim.SGD(parameters, lr=args.lr_net, momentum=0.9, weight_decay=0.0005)

    return projector, optimizer



def epoch_ssl(trainloader, net, projector, optimizer, args):
    """Run one epoch of self-supervised training (SimCLR NT-Xent or Barlow Twins) and return the mean loss.

    Each batch is augmented twice with DiffAugment to give two views. ``args.ssl_method`` selects the loss.
    Not used by MIPA.
    """
    loss_avg, num_exp = 0, 0
    net.train()
    projector.train()
    # DataParallel hides the real network behind ``.module``
    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed
    ssl_method = args.ssl_method.lower()

    for i_batch, datum in enumerate(trainloader):
        img = datum[0].float().to(args.device)
        # two independently augmented views of the same images (no seed is passed, so each call draws its own random transform)
        img_1 = DiffAugment(img.clone(), args.ssl_aug_strategy, param=args.dsa_param)
        img_2 = DiffAugment(img.clone(), args.ssl_aug_strategy, param=args.dsa_param)

        # backbone features -> projection head; the SSL loss is computed on the projections z_1, z_2 of shape (B, projection_dim)
        z_1 = projector(embed(img_1))
        z_2 = projector(embed(img_2))

        if ssl_method == 'simclr':
            # NT-Xent (SimCLR): L2-normalize the projections so that dot products are cosine similarities
            z_1 = F.normalize(z_1, dim=1)
            z_2 = F.normalize(z_2, dim=1)
            # stack both views: (2B, D) with rows 0..B-1 = view 1 and rows B..2B-1 = view 2
            z = torch.cat([z_1, z_2], dim=0)
            # (2B, 2B) cosine-similarity logits scaled by the temperature
            logits = torch.mm(z, z.t()) / args.temperature
            # a sample must not be its own negative or positive: set the diagonal to a very large negative value
            mask = torch.eye(logits.shape[0], dtype=torch.bool, device=args.device)
            logits = logits.masked_fill(mask, -1e9)
            # the positive of row i is its other view, at index (i + B) mod 2B
            targets = (torch.arange(logits.shape[0], device=args.device) + img.shape[0]) % logits.shape[0]
            loss = F.cross_entropy(logits, targets)

        elif ssl_method in ['barlowtwins', 'barlow_twins']:
            # standardize every feature over the batch (zero mean, unit std per feature)
            z_1 = (z_1 - torch.mean(z_1, dim=0)) / (torch.std(z_1, dim=0, unbiased=False) + 1e-5)
            z_2 = (z_2 - torch.mean(z_2, dim=0)) / (torch.std(z_2, dim=0, unbiased=False) + 1e-5)
            # (D, D) cross-correlation matrix between the features of the two views
            cross_correlation = torch.mm(z_1.t(), z_2) / img.shape[0]
            diagonal = torch.diagonal(cross_correlation)
            # Barlow Twins loss: pull the diagonal to 1 (invariance), push off-diagonal entries to 0 (redundancy reduction)
            on_diagonal = torch.sum((diagonal - 1)**2)
            off_diagonal = torch.sum((cross_correlation - torch.diag(diagonal))**2)
            loss = on_diagonal + args.barlow_lambda * off_diagonal

        else:
            exit('unknown SSL method: %s'%args.ssl_method)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg += loss.item() * img.shape[0]
        num_exp += img.shape[0]

    loss_avg /= num_exp

    return loss_avg



def get_transport_plan(output_real, output_syn, args):
    """Entropic optimal-transport plan between real and synthetic embeddings, computed by log-domain Sinkhorn.

    ``output_real`` is (num_real, D) and ``output_syn`` is (num_syn, D). ``args.ot_lambda`` is the entropic
    regularization and ``args.sinkhorn_iterations`` the number of iterations. Returns a (num_real, num_syn)
    plan without gradient whose marginals are approximately uniform. Not used by MIPA.
    """
    # the plan is a fixed target: no gradient flows through the Sinkhorn iterations
    with torch.no_grad():
        # L2-normalize, then use the cosine cost: cost[i, j] = 1 - cos(real_i, syn_j), shape (num_real, num_syn)
        output_real = F.normalize(output_real, dim=1)
        output_syn = F.normalize(output_syn, dim=1)
        cost = 1 - torch.mm(output_real, output_syn.t())

        num_real = output_real.shape[0]
        num_syn = output_syn.shape[0]
        # log of the uniform marginals: a = 1/num_real over real samples, b = 1/num_syn over synthetic samples
        log_a = torch.full((num_real,), -np.log(num_real), device=cost.device, dtype=cost.dtype)
        log_b = torch.full((num_syn,), -np.log(num_syn), device=cost.device, dtype=cost.dtype)
        # log of the Gibbs kernel K = exp(-cost / ot_lambda); everything stays in the log domain for stability
        log_kernel = -cost / args.ot_lambda
        # log of the Sinkhorn scaling vectors u and v, initialized to 1
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        # Sinkhorn: alternately rescale the rows (u) and the columns (v) until the plan matches the marginals a and b
        for t in range(args.sinkhorn_iterations):
            log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)

        # plan = diag(u) K diag(v); rows sum to about a and columns to about b
        transport_plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))

    return transport_plan







# Disabled earlier version of transport_contrastive_loss (OT-weighted cross-entropy of the similarity
# softmax); the active function with the same name is defined further below.
# def transport_contrastive_loss(output_real, output_syn, transport_plan, args):
#     output_real = F.normalize(output_real, dim=1)
#     output_syn = F.normalize(output_syn, dim=1)
#     logits = torch.mm(output_real, output_syn.t()) / args.temperature
#     log_probability = F.log_softmax(logits, dim=1)
#     loss = -torch.sum(transport_plan * log_probability)

#     return loss


def transport_barycentric_loss(output_real, output_syn, transport_plan, args):
    """Pull each synthetic embedding to the OT-weighted barycenter of the real embeddings assigned to it.

    Loss = mean(1 - cos(syn_j, barycenter_j)) + ``args.geometry_weight`` * mean squared difference between the
    synthetic-synthetic and barycenter-barycenter cosine similarities (off-diagonal pairs). ``output_real`` is
    (num_real, D), ``output_syn`` (num_syn, D), ``transport_plan`` (num_real, num_syn). Not used by MIPA.
    """
    # OT assignments are treated as fixed targets during the synthetic update
    transport_plan = transport_plan.detach()

    # Real representations should not receive gradients
    output_real = F.normalize(output_real.detach(), dim=1)
    output_syn = F.normalize(output_syn, dim=1)

    # ---------------------------------------------------------
    # 1. OT-weighted real barycenter for each synthetic sample
    # ---------------------------------------------------------
    # weights[i, j] = share of real sample i in the barycenter of synthetic sample j (every column sums to 1)
    weights = transport_plan / (
        transport_plan.sum(dim=0, keepdim=True).clamp_min(1e-12)
    )

    # [num_syn, dim]
    barycenter_real = torch.mm(weights.t(), output_real)
    barycenter_real = F.normalize(barycenter_real, dim=1)

    # ---------------------------------------------------------
    # 2. Each synthetic representation matches ITS barycenter
    # ---------------------------------------------------------
    bary_loss = (
        1.0 - (output_syn * barycenter_real).sum(dim=1)
    ).mean()

    # ---------------------------------------------------------
    # 3. Preserve relations between barycenters
    #
    # If two real barycenters are similar, their corresponding
    # synthetic samples should also be similar.
    # If they are far apart, synthetic samples should be too.
    # ---------------------------------------------------------
    sim_syn = torch.mm(output_syn, output_syn.t())
    sim_real = torch.mm(barycenter_real, barycenter_real.t())

    # Ignore diagonal since self-similarity is always ~1
    M = output_syn.shape[0]
    mask = ~torch.eye(M, dtype=torch.bool, device=output_syn.device)

    geom_loss = ((sim_syn - sim_real) ** 2)[mask].mean()

    # ---------------------------------------------------------
    # Final distillation loss
    # ---------------------------------------------------------
    loss = bary_loss + args.geometry_weight * geom_loss

    return loss


def transport_contrastive_loss(output_real, output_syn, transport_plan, args):
    """InfoNCE loss between synthetic embeddings and the OT barycenters of the real embeddings assigned to them.

    Synthetic sample j should be most similar to its own barycenter j among all barycenters. ``output_real``
    is (num_real, D), ``output_syn`` (num_syn, D), ``transport_plan`` (num_real, num_syn). Not used by MIPA.
    """

    # barycenter of the real embeddings per synthetic sample: plan^T @ real divided by the column mass, (num_syn, D)
    barycenter_real = torch.mm(transport_plan.t(), output_real)
    barycenter_real = barycenter_real / transport_plan.sum(dim=0).unsqueeze(1).clamp_min(1e-12)

    output_syn = F.normalize(output_syn, dim=1)
    barycenter_real = F.normalize(barycenter_real, dim=1)

    # logits[i, j] = cos(syn_i, barycenter_j) / temperature
    logits = torch.mm(output_syn, barycenter_real.t()) / args.temperature
    # the correct "class" of synthetic sample i is barycenter i (the diagonal)
    targets = torch.arange(output_syn.shape[0], device=output_syn.device)
    loss = F.cross_entropy(logits, targets)

    return loss


def evaluate_synset_SSL(it_eval, net, images_train, dst_train, testloader, args):
    """Evaluate a synthetic set by self-supervised pre-training followed by a linear probe on real labeled data.

    Stage 1 trains ``net`` and a projector with the SSL loss (SimCLR or Barlow Twins) on the unlabeled
    synthetic ``images_train``. Stage 2 freezes the backbone and trains a linear classifier on a random
    ``args.label_percentage`` % subset of the real labeled ``dst_train``. Returns (net, linear-probe train
    accuracy, test accuracy on ``testloader``). Not used by MIPA.
    """
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    # the learning rate is dropped 10x once, after epoch Epoch//2 + 1
    lr_schedule = [Epoch//2+1]
    ssl_method = args.ssl_method.lower()

    net.eval()
    # backbone embedding function (DataParallel hides it behind ``.module``); probe one image for the feature size
    embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed
    with torch.no_grad():
        feature_dim = embed(images_train[:1]).shape[1]

    # SSL projection head: feature_dim -> projection_dim -> projection_dim
    projector = nn.Sequential(
        nn.Linear(feature_dim, args.projection_dim),
        nn.ReLU(inplace=True),
        nn.Linear(args.projection_dim, args.projection_dim),
    ).to(args.device)

    # backbone and projector are trained jointly
    parameters = list(net.parameters()) + list(projector.parameters())
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0005)

    # unlabeled synthetic images only: SSL needs no labels
    dst_ssl = torch.utils.data.TensorDataset(images_train.detach().cpu())
    trainloader_ssl = torch.utils.data.DataLoader(dst_ssl, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    # Stage 1: self-supervised pre-training on the synthetic images
    for ep in range(Epoch+1):
        net.train()
        projector.train()
        loss_ssl, num_exp = 0, 0

        for i_batch, datum in enumerate(trainloader_ssl):
            img = datum[0].float().to(args.device)
            # two independently augmented views of the same images (no seed is passed, so each call draws its own random transform)
            img_1 = DiffAugment(img.clone(), args.ssl_aug_strategy, param=args.dsa_param)
            img_2 = DiffAugment(img.clone(), args.ssl_aug_strategy, param=args.dsa_param)

            # backbone features -> projection head; the SSL loss is computed on the projections z_1, z_2 of shape (B, projection_dim)
            z_1 = projector(embed(img_1))
            z_2 = projector(embed(img_2))

            if ssl_method == 'simclr':
                # NT-Xent (SimCLR): L2-normalize the projections so that dot products are cosine similarities
                z_1 = F.normalize(z_1, dim=1)
                z_2 = F.normalize(z_2, dim=1)
                # stack both views: (2B, D) with rows 0..B-1 = view 1 and rows B..2B-1 = view 2
                z = torch.cat([z_1, z_2], dim=0)
                # (2B, 2B) cosine-similarity logits scaled by the temperature
                logits = torch.mm(z, z.t()) / args.temperature
                # a sample must not be its own negative or positive: set the diagonal to a very large negative value
                mask = torch.eye(logits.shape[0], dtype=torch.bool, device=args.device)
                logits = logits.masked_fill(mask, -1e9)
                # the positive of row i is its other view, at index (i + B) mod 2B
                targets = (torch.arange(logits.shape[0], device=args.device) + img.shape[0]) % logits.shape[0]
                loss = F.cross_entropy(logits, targets)

            elif ssl_method in ['barlowtwins', 'barlow_twins']:
                # standardize every feature over the batch (zero mean, unit std per feature)
                z_1 = (z_1 - torch.mean(z_1, dim=0)) / (torch.std(z_1, dim=0, unbiased=False) + 1e-5)
                z_2 = (z_2 - torch.mean(z_2, dim=0)) / (torch.std(z_2, dim=0, unbiased=False) + 1e-5)
                # (D, D) cross-correlation matrix between the features of the two views
                cross_correlation = torch.mm(z_1.t(), z_2) / img.shape[0]
                diagonal = torch.diagonal(cross_correlation)
                # Barlow Twins loss: pull the diagonal to 1 (invariance), push off-diagonal entries to 0 (redundancy reduction)
                on_diagonal = torch.sum((diagonal - 1)**2)
                off_diagonal = torch.sum((cross_correlation - torch.diag(diagonal))**2)
                loss = on_diagonal + args.barlow_lambda * off_diagonal

            else:
                exit('unknown SSL method: %s'%args.ssl_method)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ssl += loss.item() * img.shape[0]
            num_exp += img.shape[0]

        loss_ssl /= num_exp

        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Stage 2 (linear probe): freeze the backbone; only a new linear classifier on its features is trained
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    with torch.no_grad():
        # number of classes = output size of the network's own classifier
        num_classes = net(images_train[:1]).shape[1]

    # labeled subset of the REAL training set: label_percentage % of its samples, at least one
    num_labeled = max(1, int(len(dst_train) * args.label_percentage / 100))
    # seeded with args.seed (default 0) + it_eval: a different but reproducible subset for each evaluation run
    indices_labeled = np.random.RandomState(getattr(args, 'seed', 0) + it_eval).permutation(len(dst_train))[:num_labeled]
    dst_labeled = torch.utils.data.Subset(dst_train, indices_labeled)
    trainloader_linear = torch.utils.data.DataLoader(dst_labeled, batch_size=args.batch_linear, shuffle=True, num_workers=0)

    # linear classifier on the frozen feature_dim-dimensional embedding
    linear = nn.Linear(feature_dim, num_classes).to(args.device)
    lr_linear = float(args.lr_linear)
    Epoch_linear = int(args.epoch_linear_train)
    lr_schedule_linear = [Epoch_linear//2+1]
    optimizer_linear = torch.optim.SGD(linear.parameters(), lr=lr_linear, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    def epoch_linear(mode, dataloader):
        """One pass of the linear probe over ``dataloader``; returns (mean loss, accuracy). The backbone is frozen."""
        loss_avg, acc_avg, num_exp = 0, 0, 0

        if mode == 'train':
            linear.train()
        else:
            linear.eval()

        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(args.device)
            lab = datum[1].long().to(args.device)
            n_b = lab.shape[0]

            # the backbone is frozen: features are computed without gradients and only ``linear`` is updated
            with torch.no_grad():
                feature = embed(img)
            output = linear(feature.detach())
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b

            if mode == 'train':
                optimizer_linear.zero_grad()
                loss.backward()
                optimizer_linear.step()

        loss_avg /= num_exp
        acc_avg /= num_exp

        return loss_avg, acc_avg

    for ep in range(Epoch_linear+1):
        loss_train, acc_train = epoch_linear('train', trainloader_linear)
        if ep in lr_schedule_linear:
            lr_linear *= 0.1
            optimizer_linear = torch.optim.SGD(linear.parameters(), lr=lr_linear, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    # the linear probe is evaluated on the real test set
    loss_test, acc_test = epoch_linear('test', testloader)
    print('%s Evaluate_SSL_%02d: method = %s ssl epoch = %04d linear epoch = %04d labeled = %.2f%% train time = %d s ssl loss = %.6f train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, args.ssl_method, Epoch, Epoch_linear, args.label_percentage, int(time_train), loss_ssl, loss_train, acc_train, acc_test))

    return net, acc_train, acc_test



def augment(images, dc_aug_param, device):
    """DC-style random augmentation applied in place to ``images`` (N, C, H, W); returns the same tensor.

    Every image independently gets ONE operation chosen at random from ``dc_aug_param['strategy']``, an
    underscore-separated subset of 'crop', 'scale', 'rotate', 'noise'. The other keys of ``dc_aug_param`` are
    'crop' (max shift in pixels), 'scale' and 'rotate' (max relative scale change and rotation in degrees)
    and 'noise' (std of Gaussian noise). Not differentiable; MIPA uses DiffAugment instead. Not used by MIPA.
    """
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        # per-channel mean of the whole batch, used as the fill value for padded and rotated borders
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            """Random translation of image i: pad by ``crop`` pixels (channel means) and crop back to the original size."""
            # canvas of shape (C, H + 2*crop, W + 2*crop), filled with the channel means below
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            # random top-left corner of the crop window, each in [0, 2*crop)
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            """Random rescale of image i, then re-center it on the original canvas (cropped if larger, zero-padded if smaller).

            Both scale factors are drawn from [1 - scale, 1 + scale] independently, but both are applied to
            ``shape[2]`` (the height), which is only right for square images.
            """
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            # resize to (h, w); F.interpolate defaults to nearest-neighbour
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            # square canvas large enough for both the scaled and the original image
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            # paste the scaled image into the center of the (zero-filled) canvas
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            # cut the central window of the original size out of the canvas
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            """Random rotation of image i by an integer angle in [-rotate, rotate) degrees, using scipy."""
            # rotate in the (H, W) plane; scipy enlarges the array by default (reshape=True), borders get the mean value
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            # crop the central window of the original size out of the enlarged rotated image
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            """Add Gaussian noise with standard deviation ``noise`` to image i."""
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        # e.g. 'crop_scale_rotate' -> ['crop', 'scale', 'rotate']
        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    """Return the DC augmentation parameters (crop 4, scale 0.2, rotate 45, noise 0.001) for ``augment``.

    The strategy is 'none' except 'crop_scale_rotate' for MNIST and 'crop_noise' when evaluating ConvNetBN;
    ``model`` and ``ipc`` are unused. Not used by MIPA.
    """
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    """Return the list of architecture names used to evaluate a synthetic set for ``eval_mode``.

    'M': several architectures, 'B': several BatchNorm architectures, 'W'/'D'/'A'/'P'/'N': ablations over
    width/depth/activation/pooling/normalization, 'S': the training model itself (BN replaced by IN), 'SS':
    the training model unchanged; anything else gives [model_eval]. Not used by MIPA.
    """
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        # drop the 'BN' suffix, e.g. 'ConvNetBN' -> 'ConvNet' (instance-norm version)
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    """Hyper-parameters (and per-call state) of the differentiable Siamese augmentation (DSA).

    One instance is passed to ``DiffAugment`` and on to every ``rand_*`` function. ``DiffAugment``
    additionally sets ``Siamese`` and ``latestseed`` on it at call time. MIPA uses ``ParamDiffAug()`` with
    the defaults below (dm_dsa.make_diff_augmenter also sets ``aug_mode = 'S'``, which is the default).
    """
    def __init__(self):
        """Set the default augmentation ranges (see the comment above each attribute)."""
        # 'S' (single): apply ONE randomly chosen group of the strategy per call; 'M' (multiple): apply every group
        self.aug_mode = 'S' #'multiple or single'
        # probability of a horizontal flip, per image
        self.prob_flip = 0.5
        # scale factors are drawn from [1/1.2, 1.2] (zoom in or out)
        self.ratio_scale = 1.2
        # maximum rotation in degrees: the angle is drawn from [-15, 15]
        self.ratio_rotate = 15.0
        # maximum translation as a fraction of the image size (0.125 * 32 = 4 pixels)
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        # brightness offset is (u - 0.5) * brightness with u in [0, 1], i.e. within +-0.5
        self.brightness = 1.0
        # saturation gain is u * saturation with u in [0, 1], i.e. within [0, 2]
        self.saturation = 2.0
        # contrast gain is u + contrast with u in [0, 1], i.e. within [0.5, 1.5]
        self.contrast = 0.5


def set_seed_DiffAug(param):
    """Reseed torch's global RNG with ``param.latestseed`` and advance it; do nothing if it is -1.

    Called before every random draw in the augmentations, so a given seed reproduces the same transform.
    Because this reseeds the global RNG, dm_dsa.make_diff_augmenter runs DiffAugment inside
    ``torch.random.fork_rng`` to avoid side effects.
    """
    if param.latestseed == -1:
        return
    else:
        # the seed is advanced after each use so that successive draws differ but stay reproducible
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    """Apply a differentiable augmentation strategy to a batch of images; gradients flow back to ``x``.

    Args:
        x: Float tensor of shape (N, C, H, W); in MIPA, images in dataset-normalized space.
        strategy: Underscore-separated augmentation groups from AUGMENT_FNS, e.g. 'color_crop_flip'.
            '', 'none' and 'None' return ``x`` unchanged.
        seed: -1 draws an independent random transform per image. Any other value enables Siamese mode:
            the draws are seeded with it and the first image's random parameters are copied to the whole
            batch, so all images get the same transform and the same seed reproduces it.
        param: ParamDiffAug with the ranges and ``aug_mode``; DiffAugment sets ``param.Siamese`` and
            ``param.latestseed`` on it.

    Returns:
        The augmented images, same shape as ``x`` (made contiguous).

    Used by MIPA (distill.py through dm_dsa.py).
    """
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    # a given seed switches on Siamese mode: one shared random transform for the whole batch
    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    # consumed (and advanced) by set_seed_DiffAug before every random draw; -1 means "do not reseed"
    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            # aug_mode 'M': apply every group of the strategy, in the listed order
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            # aug_mode 'S': apply only ONE group, chosen uniformly at random; the pick is made after reseeding, so a given seed gives the same pick
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        # some augmentations (e.g. rand_crop) return permuted, non-contiguous views
        x = x.contiguous()
    return x


# Differentiable augmentations, all with the signature f(x, param) -> x for x of shape (N, C, H, W).
# Random parameters are drawn per image; in Siamese mode (param.Siamese) the first image's parameters are
# copied to all images, so the whole batch gets the same transform.
def rand_scale(x, param):
    """Random zoom of each image by independent factors sx, sy in [1/ratio_scale, ratio_scale] (affine grid).

    Uses bilinear resampling; regions that fall outside the original image are filled with zeros.
    Output has the same shape as ``x``.
    """
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    # sx, sy are uniform in [1/ratio, ratio]; sx scales the x (width) axis and sy the y (height) axis
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    # (N, 2, 3) affine matrices [[sx, 0, 0], [0, sy, 0]]: scaling about the image center (theta maps output to input coordinates, so a factor below 1 enlarges the image)
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    # (N, H, W, 2) sampling grid in [-1, 1] coordinates (theta was built on the CPU, so move the grid to x's device)
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    """Random rotation of each image by an angle uniform in [-ratio_rotate, +ratio_rotate] degrees (affine grid).

    Bilinear resampling about the image center; corners that rotate in from outside are filled with zeros.
    """
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    # random angle in radians, uniform in [-ratio, +ratio] degrees
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    # (N, 2, 3) rotation matrices [[cos, -sin, 0], [sin, cos, 0]]
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
    x = F.grid_sample(x, grid, align_corners=False)
    return x


def rand_flip(x, param):
    """Flip each image horizontally (along the width axis) with probability ``prob_flip``."""
    prob = param.prob_flip
    set_seed_DiffAug(param)
    # one uniform draw per image, shape (N, 1, 1, 1) so that it broadcasts over C, H, W
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0]
    # flip the images whose draw is below prob_flip and keep the others
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    """Add a random brightness offset (u - 0.5) * ``brightness`` with u in [0, 1], one value per image."""
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    # additive shift in [-ratio/2, +ratio/2], the same for every pixel and channel of an image
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    """Randomly scale each pixel's deviation from its gray level (mean over channels): x' = (x - gray) * (u * ratio) + gray."""
    ratio = param.saturation
    # (N, 1, H, W) mean over channels at each pixel, i.e. a gray-level image
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    # gain u * ratio (u in [0, 1]) between the gray image and x; a gain above 1 over-saturates
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    """Randomly scale each image's deviation from its own mean: x' = (x - mean) * (u + ratio) + mean."""
    ratio = param.contrast
    # (N, 1, 1, 1) mean over all channels and pixels of each image
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    # gain u + ratio (u in [0, 1]) around the image mean
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    """Randomly translate each image by up to ``ratio_crop_pad`` of its size per axis; exposed borders are zeros.

    Implemented as padding by one zero pixel and gathering with integer index shifts (no resampling).
    """
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    # maximum shift in pixels along the height (x, dim 2) and the width (y, dim 3), rounded to nearest
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    # integer shift per image, shape (N, 1, 1) so that it broadcasts over the (N, H, W) index grid
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    # index grids of shape (N, H, W) for the batch index, the row and the column
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij',
    )
    # source row/column of every output pixel; +1 accounts for the one-pixel zero border added below, and clamping sends out-of-range pixels onto that border
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    # one-pixel zero border around H and W
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    # gather in channels-last layout with advanced indexing, then return to (N, C, H, W)
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    """Zero out a square box of side ``ratio_cutout`` * image size at a random position in each image.

    The box may stick out of the image; the part outside is clipped.
    """
    ratio = param.ratio_cutout
    # box height and width in pixels
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    # random box center per image; the (1 - size % 2) term adjusts the range of centers for odd box sizes
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    # index grids (N, box_h, box_w) enumerating the pixels of one box per image
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij',
    )
    # absolute pixel coordinates of the box (centered on the offset), clamped into the image
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    # (N, H, W) mask of ones with zeros inside each image's box; multiplied into x with a channel axis added
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


# Maps each strategy token accepted by DiffAugment to the augmentations it triggers; the 'color' group
# applies brightness, saturation and contrast together. distill.py validates its strategy against these keys.
AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}




def transport_soft_assignment_loss(
    output_real,
    output_syn,
    transport_plan,
    args
):
    """Match the similarity-based softmax over synthetic images to the OT assignment of each real image.

    For real image i, q(. | i) is the row-normalized transport plan and p(. | i) = softmax_j(cos(real_i, syn_j) /
    temperature); the loss is the cross-entropy -sum_j q_ij log p_ij averaged over real images. ``output_real``
    is (num_real, D), ``output_syn`` (num_syn, D), ``transport_plan`` (num_real, num_syn). Not used by MIPA.
    """
    # OT is a fixed target during this image update
    q = transport_plan.detach()

    # Normalize features
    output_real = F.normalize(output_real.detach(), dim=1)
    output_syn = F.normalize(output_syn, dim=1)

    # ---------------------------------------------------------
    # q_ij = OT assignment distribution over synthetic images
    # for each real image i
    # ---------------------------------------------------------
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # ---------------------------------------------------------
    # p_ij = similarity-induced distribution
    # ---------------------------------------------------------
    logits = torch.mm(output_real, output_syn.t()) / args.temperature

    log_p = F.log_softmax(logits, dim=1)

    # Match p(s_j | x_i) to OT assignment q(s_j | x_i)
    loss = -(q * log_p).sum(dim=1).mean()

    return loss
