"""Backbone networks for the dataset-distillation benchmarks (MLP, ConvNet, LeNet, AlexNet, VGG, ResNet).

Used by MIPA: common.py imports this module as ``networks`` and builds the frozen teacher encoder
in ``common.build_encoder`` from these pieces only:
    * ResNet18 (and ``ResNet`` with ``BasicBlock``/``Bottleneck`` for the 34/50-layer variants),
    * VGG11BN,
    * ConvNet.
MIPA always selects batch normalization (``norm='batchnorm'`` / ``net_norm='batchnorm'``), replaces the
``classifier`` head with ``nn.Identity`` so that ``forward`` returns the pooled representation, and then
re-initializes the weights (it relies on the attribute names ``bn2`` of BasicBlock and ``bn3`` of
Bottleneck, so do not rename them).

Everything else here (MLP, ConvNetGAP, LeNet, AlexNet, the average-pool ``*_AP`` ResNets, ResNetImageNet,
the other VGG/ResNet factory functions) is legacy benchmark code that MIPA does not use.

Conventions shared by all networks:
    * Inputs are float tensors of shape (N, channel, H, W); 1-channel inputs are assumed to be 28x28 and
      are padded up to 32x32 where a layer needs it.
    * ``norm='instancenorm'`` is implemented as ``nn.GroupNorm(C, C)`` (one group per channel), which is
      equivalent to instance normalization with a learnable affine; any other value of ``norm`` in the
      ResNet/VGG classes falls back to ``nn.BatchNorm2d``.
    * ``classifier`` is the final ``nn.Linear`` head (``num_classes`` outputs).
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


''' MLP '''
class MLP(nn.Module):
    """Three-layer fully connected classifier (input -> 128 -> 128 -> num_classes) on flattened images.

    The input size is fixed: 28*28*1 for 1-channel data, otherwise 32*32*3. Not used by MIPA.
    """
    def __init__(self, channel, num_classes):
        """Create the layers of the network.

        Args:
            channel: Number of input channels; 1 selects a 28*28 input, anything else 32*32*3.
            num_classes: Number of output logits.
        """
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Flatten ``x`` of shape (N, C, H, W) and return logits of shape (N, num_classes)."""
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out



''' ConvNet '''
class ConvNet(nn.Module):
    """Plain convolutional network of ``net_depth`` identical blocks (the standard distillation ConvNet).

    Each block is Conv3x3 -> normalization -> activation -> optional 2x2 pooling, with ``net_width``
    channels everywhere. A single linear classifier acts on the flattened last feature map (there is no
    global pooling), so the classifier input size is net_width * H' * W' and depends on the image size.

    Used by MIPA as the 'ConvNet' teacher backbone (common.build_encoder builds it with 128 channels,
    depth 3 or 4, 'relu', 'batchnorm', 'avgpooling' and replaces ``classifier`` with nn.Identity, so the
    forward pass returns the flattened feature map of shape (N, net_width * H' * W')).
    """
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        """Create the layers of the network.

        Args:
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
            net_width: Number of channels of every conv layer.
            net_depth: Number of conv blocks.
            net_act: 'sigmoid', 'relu' or 'leakyrelu'.
            net_norm: 'batchnorm', 'layernorm', 'instancenorm', 'groupnorm' or 'none'.
            net_pooling: 'maxpooling', 'avgpooling' or 'none'; 2x2 with stride 2, applied after every block.
            im_size: (H, W) of the input images; used to size the classifier and, for layernorm, the norm layers.
        """
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        # size of the flattened final feature map: channels * height * width (shape_feat comes from _make_layers)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        """Return logits of shape (N, num_classes) for images ``x`` of shape (N, channel, H, W)."""
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        # flatten (N, C', H', W') to (N, C' * H' * W')
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        """Return a new activation module for the name ``net_act`` (exits on an unknown name)."""
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        """Return a 2x2, stride-2 pooling layer for ``net_pooling``, or None for 'none' (exits if unknown)."""
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        """Return the normalization layer for a feature map of shape ``shape_feat`` = (C, H, W), or None.

        'layernorm' normalizes jointly over (C, H, W), which is why the spatial size is needed here.
        'instancenorm' is GroupNorm with one group per channel; 'groupnorm' uses 4 groups.
        """
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            # normalizes over the whole (C, H, W) map with an elementwise affine of the same shape
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            # one group per channel == instance normalization
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        """Build the conv blocks and track the shape of the final feature map.

        Returns:
            (nn.Sequential of all layers, shape_feat) where shape_feat = [C, H, W] is the per-sample shape of
            the output of the last block (without the batch dimension).
        """
        layers = []
        in_channels = channel
        # 28x28 inputs are treated as 32x32 for the shape bookkeeping below: the first conv pads by 3 for
        # 1-channel inputs, which turns 28x28 into 32x32 (28 + 2*3 - 3 + 1 = 32)
        if im_size[0] == 28:
            im_size = (32, 32)
        # running (C, H, W) shape, updated block by block; used for norm layers and the classifier size
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            # 3x3 conv; padding=3 on the first layer of 1-channel (28x28) inputs, otherwise 1 (shape preserving)
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                # 2x2 pooling with stride 2 halves height and width (floor division)
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat


''' ConvNet '''
class ConvNetGAP(nn.Module):
    """ConvNet variant with global average pooling in front of the classifier.

    The conv blocks are identical to ConvNet, but the last feature map is reduced to (N, net_width, 1, 1) by
    adaptive average pooling, so the classifier input is ``net_width`` and does not depend on the image
    size. Not used by MIPA.
    """
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        """Same arguments as ConvNet.__init__; the classifier maps ``net_width`` features to ``num_classes``."""
        super(ConvNetGAP, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        # num_feat is not used with global average pooling (kept from the flatten-based ConvNet)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        # self.classifier = nn.Linear(num_feat, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(shape_feat[0], num_classes)

    def forward(self, x):
        """Return logits (N, num_classes): conv blocks -> global average pooling -> linear classifier."""
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        """Same as ConvNet._get_activation."""
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        """Same as ConvNet._get_pooling."""
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        """Same as ConvNet._get_normlayer; ``shape_feat`` is (C, H, W)."""
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        """Same as ConvNet._make_layers; returns (nn.Sequential, shape_feat) with shape_feat = [C, H, W]."""
        layers = []
        in_channels = channel
        # 28x28 inputs are padded to 32x32 by the first conv (padding=3 for 1-channel inputs below)
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            # padding=3 on the first layer of 1-channel inputs grows 28x28 to 32x32, otherwise padding=1
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' LeNet '''
class LeNet(nn.Module):
    """LeNet-5-style CNN: two 5x5 conv + max-pool stages, then fully connected layers 400 -> 120 -> 84 -> num_classes.

    The first conv pads by 2 for 1-channel (28x28) inputs and by 0 for 3-channel (32x32) inputs, so both end
    with a 16 x 5 x 5 feature map. Not used by MIPA.
    """
    def __init__(self, channel, num_classes):
        """Create the layers of the network.

        Args:
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
        """
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, 28 or 32, 28 or 32)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x



''' AlexNet '''
class AlexNet(nn.Module):
    """AlexNet-style CNN for small images: five conv layers, three 2x2 max-pools and a linear classifier.

    The last feature map has 192 channels at 4x4 resolution for 32x32 inputs (1-channel 28x28 inputs are
    padded up to 32x32 by the first conv). Not used by MIPA.
    """
    def __init__(self, channel, num_classes):
        """Create the layers of the network.

        Args:
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
        """
        super(AlexNet, self).__init__()
        # first conv uses padding=4 for 1-channel (28x28) inputs, which makes them 32x32; padding=2 otherwise
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 32x32 input -> three 2x2 max-pools -> a 192-channel 4x4 map, flattened to 192*4*4 features
        self.classifier = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, H, W) with H, W in {28, 32}."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



''' VGG '''
# VGG layer configurations: an integer is the output width of a 3x3 conv (+ norm + ReLU), 'M' is a 2x2 max-pool
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    """VGG feature extractor with a single linear classifier, for 32x32 inputs (64x64 is also handled in forward).

    Every conv is 3x3 followed by normalization (instance norm as GroupNorm, or batch norm) and ReLU. The
    five 'M' entries of the VGG11 config reduce 32x32 to 1x1, so the classifier sees the 512 channels of
    the last conv.

    Used by MIPA through the factory function VGG11BN (batch-norm VGG11 teacher backbone).
    """
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            vgg_name: Key of ``cfg_vgg`` ('VGG11', 'VGG13', 'VGG16' or 'VGG19').
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
            norm: 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        # 512 = channels of the last conv (1x1 spatial after pooling); the 128-feature 'VGGS' branch is never
        # reached because cfg_vgg has no 'VGGS' entry
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, 32, 32) (or 64x64, see below)."""
        x = self.features(x)
        # 32x32 already yields 1x1. Also support native 64x64 Tiny ImageNet
        # without changing the classifier dimension or the 32x32 behavior.
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, norm):
        """Build the conv/norm/ReLU stack described by ``cfg`` (integers = conv width, 'M' = max-pool).

        ``norm`` is 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # padding=3 on the very first conv of 1-channel (28x28) inputs makes them 32x32, otherwise padding=1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # 1x1 average pooling with stride 1 is an identity op (kept from the original benchmark code)
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(channel, num_classes):
    """VGG11 with instance norm (GroupNorm). Not used by MIPA."""
    return VGG('VGG11', channel, num_classes)
def VGG11BN(channel, num_classes):
    """VGG11 with batch norm. Used by MIPA as the 'VGG11' teacher backbone (common.build_encoder)."""
    return VGG('VGG11', channel, num_classes, norm='batchnorm')
def VGG13(channel, num_classes):
    """VGG13 with instance norm. Not used by MIPA."""
    return VGG('VGG13', channel, num_classes)
def VGG16(channel, num_classes):
    """VGG16 with instance norm. Not used by MIPA."""
    return VGG('VGG16', channel, num_classes)
def VGG19(channel, num_classes):
    """VGG19 with instance norm. Not used by MIPA."""
    return VGG('VGG19', channel, num_classes)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=2, stride=2)

class BasicBlock_AP(nn.Module):
    """ResNet basic block (two 3x3 convs + skip) that downsamples with average pooling instead of a strided conv.

    For stride 2, conv(stride=2) is replaced by conv(stride=1) followed by 2x2 average pooling, both on the
    main path and in the projection shortcut. Not used by MIPA.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels (``expansion`` is 1, so output channels equal ``planes``).
            stride: 1 keeps the resolution, 2 halves it (by average pooling).
            norm: 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # projection shortcut (1x1 conv + pool + norm) when the shape changes, otherwise an identity shortcut
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """Return relu(main_path(x) + shortcut(x)); height and width are divided by ``stride``."""
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    """ResNet bottleneck block (1x1 reduce, 3x3, 1x1 expand by 4, + skip) with average-pool downsampling.

    For stride 2 the 3x3 conv keeps stride 1 and a 2x2 average pooling is applied before the expanding 1x1
    conv (and in the projection shortcut). Not used by MIPA.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            in_planes: Number of input channels.
            planes: Width of the bottleneck; the block outputs ``expansion * planes`` = 4 * planes channels.
            stride: 1 keeps the resolution, 2 halves it (by average pooling).
            norm: 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        # projection shortcut (1x1 conv + pool + norm) when the shape changes, otherwise an identity shortcut
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """Return relu(main_path(x) + shortcut(x)); height and width are divided by ``stride``."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    """CIFAR-style ResNet built from the average-pool blocks, with a classifier on the flattened last feature map.

    Stem is a 3x3 stride-1 conv (no max-pool); the four stages have 64/128/256/512 base channels and strides
    1/2/2/2 (so 32x32 inputs end as 4x4 maps). Not used by MIPA.
    """
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            block: BasicBlock_AP or Bottleneck_AP (class, not instance).
            num_blocks: Number of blocks in each of the four stages, e.g. [2, 2, 2, 2].
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
            norm: 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # unlike ResNet there is no global pooling: the flattened last map (4x4 for 32x32 inputs, 3x3 for
        # 1-channel 28x28 inputs since 28 -> 14 -> 7 -> 3) goes straight into the classifier
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        """Stack ``num_blocks`` blocks; only the first one uses ``stride``, the others use stride 1.

        Also updates ``self.in_planes`` so the next stage sees the right input width.
        """
        # the first block of a stage may downsample, the remaining blocks keep the resolution
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            # the next block's input width is the output width of this one
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, H, W) with H = W = 32 (or 28 for 1-channel)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 1x1 average pooling is an identity op; the feature map is flattened as is
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18BN_AP(channel, num_classes):
    """ResNet-18 with average-pool blocks and batch norm. Not used by MIPA."""
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    """ResNet-18 with average-pool blocks and instance norm. Not used by MIPA."""
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs with normalization and a skip connection.

    Used by MIPA: it is the block inside ResNet18 and the 34-layer encoder, and common.build_encoder also
    refers to ``networks.BasicBlock`` directly to zero-initialize ``bn2.weight`` of every block, so the
    attribute names must stay as they are.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels (``expansion`` is 1, so output channels equal ``planes``).
            stride: Stride of the first 3x3 conv (and of the projection shortcut); 2 halves the resolution.
            norm: 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # projection shortcut (1x1 conv + norm) when the shape changes, otherwise an identity shortcut
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """Return relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # residual connection: the (identity or projected) input is added before the final ReLU
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet bottleneck block: 1x1 reduce, 3x3 (carries the stride), 1x1 expand to 4 * planes, plus a skip connection.

    Used by MIPA: it is the block of the 50-layer encoder, and common.build_encoder also refers to
    ``networks.Bottleneck`` directly to zero-initialize ``bn3.weight`` of every block, so the attribute
    names must stay as they are.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            in_planes: Number of input channels.
            planes: Width of the bottleneck; the block outputs ``expansion * planes`` = 4 * planes channels.
            stride: Stride of the 3x3 conv (and of the projection shortcut); 2 halves the resolution.
            norm: 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        # projection shortcut (1x1 conv + norm) when the shape changes, otherwise an identity shortcut
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """Return relu(bn3(conv3(relu(bn2(conv2(relu(bn1(conv1(x)))))))) + shortcut(x))."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """CIFAR-style ResNet: 3x3 stride-1 stem (no max-pool), four stages and a linear classifier.

    The stages have 64/128/256/512 base channels (times ``block.expansion``) with strides 1/2/2/2, followed
    by global average pooling, so the feature size is 512 * block.expansion for any input size (32x32 or
    64x64).

    Used by MIPA: the ResNet18/34/50 teacher encoders are instances of this class with norm='batchnorm'.
    common.build_encoder replaces ``classifier`` with nn.Identity, so ``forward`` then returns the pooled
    (N, 512 * expansion) representation.
    """
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            block: BasicBlock or Bottleneck (class, not instance).
            num_blocks: Number of blocks in each of the four stages, e.g. [2, 2, 2, 2] for ResNet-18.
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
            norm: 'instancenorm' (GroupNorm, one group per channel) or anything else for BatchNorm2d.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Stack ``num_blocks`` blocks; only the first one uses ``stride``, the others use stride 1.

        Also updates ``self.in_planes`` so the next stage sees the right input width.
        """
        # the first block of a stage may downsample, the remaining blocks keep the resolution
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            # the next block's input width is the output width of this one
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, H, W)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Global pooling preserves the existing 32x32 behavior while also
        # producing 512 features for Tiny ImageNet's 64x64 inputs.
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    

    def features(self, x):
        """Return the globally pooled feature map of shape (N, 512 * expansion, 1, 1), before flattening and the classifier."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return out


class ResNetImageNet(nn.Module):
    """ResNet with an ImageNet-style stem: 7x7 stride-2 conv followed by a 3x3 stride-2 max-pool.

    This downsamples the input by 4x before the first stage, so it is meant for large (ImageNet-sized)
    images. Global average pooling then gives a 512 * block.expansion feature vector. Not used by MIPA.
    """
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        """Same arguments as ResNet.__init__ (block, num_blocks, channel, num_classes, norm)."""
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Same as ResNet._make_layer: ``stride`` applies to the first block only; updates ``self.in_planes``."""
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, H, W)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def ResNet10(channel, num_classes):
    """ResNet-10 (BasicBlock, [1, 1, 1, 1]) with batch norm. Not used by MIPA."""
    return ResNet(BasicBlock, [1,1,1,1], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18(channel, num_classes): # Batchnorm Added
    """ResNet-18 (BasicBlock, [2, 2, 2, 2]) with batch norm.

    Used by MIPA as the 'ResNet18' teacher backbone (common.build_encoder).
    """
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet34(channel, num_classes):
    """ResNet-34 (BasicBlock, [3, 4, 6, 3]) with the default instance norm.

    common.build_encoder builds its 34-layer encoder through ``ResNet(...)`` with norm='batchnorm'
    instead of calling this function. Not used by MIPA.
    """
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet50(channel, num_classes):
    """ResNet-50 (Bottleneck, [3, 4, 6, 3]) with the default instance norm.

    common.build_encoder builds its 50-layer encoder through ``ResNet(...)`` with norm='batchnorm'
    instead of calling this function. Not used by MIPA.
    """
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet101(channel, num_classes):
    """ResNet-101 (Bottleneck, [3, 4, 23, 3]) with instance norm. Not used by MIPA."""
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

def ResNet152(channel, num_classes):
    """ResNet-152 (Bottleneck, [3, 8, 36, 3]) with instance norm. Not used by MIPA."""
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)

def ResNet18ImageNet(channel, num_classes):
    """ResNet-18 with the ImageNet-style stem and instance norm. Not used by MIPA."""
    return ResNetImageNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet6ImageNet(channel, num_classes):
    """ImageNet-style ResNet with a [1, 1, 1, 1] BasicBlock layout and instance norm.

    The name says 6, but this is the same block layout as ResNet10. Not used by MIPA.
    """
    return ResNetImageNet(BasicBlock, [1,1,1,1], channel=channel, num_classes=num_classes)
