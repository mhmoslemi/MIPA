"""Older copy of the benchmark backbones; imported only by utils.py (``utils.get_network``). Not used by MIPA.

MIPA builds its encoders from benchmark_networks.py instead. Python still executes this file when
utils.py is imported (utils.py imports it at module level), but none of its classes are used by MIPA.

Compared with benchmark_networks.py, this copy
    * adds a Swish activation and the 'swish' option of ConvNet,
    * adds AlexNetBN (AlexNet with batch norm) and the ``embed(x)`` method on ConvNet, AlexNet, AlexNetBN,
      VGG, ResNet_AP and ResNet, which returns the flattened features before the classifier (the
      self-supervised helpers in utils.py call ``net.embed``),
    * names the AlexNet head ``fc`` (``classifier`` in benchmark_networks.py),
    * uses a fixed ``F.avg_pool2d(out, 4)`` in ResNet, which assumes a 4x4 last feature map (32x32 inputs),
      whereas benchmark_networks.py uses adaptive pooling,
    * has no ConvNetGAP, ResNetImageNet or ResNet10, and its ResNet18 uses instance norm (the
      benchmark_networks.py ResNet18 uses batch norm).

Conventions shared by all networks:
    * Inputs are float tensors of shape (N, channel, H, W); 1-channel inputs are assumed to be 28x28 and
      are padded up to 32x32 where a layer needs it.
    * ``norm='instancenorm'`` is implemented as ``nn.GroupNorm(C, C)`` (one group per channel), which is
      equivalent to instance normalization with a learnable affine; any other value of ``norm`` in the
      ResNet/VGG classes falls back to ``nn.BatchNorm2d``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    """Swish activation, x * sigmoid(x) (also known as SiLU), applied elementwise. Not used by MIPA."""
    def __init__(self):
        """Swish has no parameters."""
        super().__init__()

    def forward(self, input):
        """Return ``input * sigmoid(input)``; the output has the same shape as the input."""
        return input * torch.sigmoid(input)


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



# ''' ConvNet '''
class ConvNet(nn.Module):
    """Plain convolutional network of ``net_depth`` identical blocks (the standard distillation ConvNet).

    Each block is Conv3x3 -> normalization -> activation -> optional 2x2 pooling, with ``net_width``
    channels everywhere. A single linear classifier acts on the flattened last feature map (there is no
    global pooling), so the classifier input size is net_width * H' * W' and depends on the image size.
    ``embed`` returns that flattened map. Not used by MIPA.
    """
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        """Create the layers of the network.

        Args:
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
            net_width: Number of channels of every conv layer.
            net_depth: Number of conv blocks.
            net_act: 'sigmoid', 'relu', 'leakyrelu' or 'swish'.
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
        out = self.features(x)
        # flatten (N, C', H', W') to (N, C' * H' * W')
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        """Return the flattened conv feature map, shape (N, net_width * H' * W'); used as the SSL representation in utils.py."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        """Return a new activation module for the name ``net_act`` (exits on an unknown name)."""
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
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




# Disabled alternative ConvNet (commented out; the active ConvNet above is the one in use). It builds the
# same layers, but embed() applies global average pooling (adaptive_avg_pool2d to 1x1), so the
# representation has ``feature_dim = shape_feat[0]`` (= net_width) entries and the classifier input
# does not depend on the image size; forward() is classifier(embed(x)).
# class ConvNet(nn.Module):
#     def __init__(
#         self,
#         channel,
#         num_classes,
#         net_width,
#         net_depth,
#         net_act,
#         net_norm,
#         net_pooling,
#         im_size=(32, 32)
#     ):
#         super(ConvNet, self).__init__()

#         self.features, shape_feat = self._make_layers(
#             channel,
#             net_width,
#             net_depth,
#             net_norm,
#             net_act,
#             net_pooling,
#             im_size
#         )

#         # after global average pooling, representation dim = channels
#         self.feature_dim = shape_feat[0]

#         self.classifier = nn.Linear(
#             self.feature_dim,
#             num_classes
#         )

#     def embed(self, x):
#         out = self.features(x)

#         # IMPORTANT:
#         # convert spatial feature map into one image-level representation
#         out = F.adaptive_avg_pool2d(out, output_size=1)

#         out = out.view(out.size(0), -1)

#         return out

#     def forward(self, x):
#         out = self.embed(x)
#         out = self.classifier(out)
#         return out

#     def _get_activation(self, net_act):
#         if net_act == 'sigmoid':
#             return nn.Sigmoid()
#         elif net_act == 'relu':
#             return nn.ReLU(inplace=True)
#         elif net_act == 'leakyrelu':
#             return nn.LeakyReLU(negative_slope=0.01)
#         elif net_act == 'swish':
#             return Swish()
#         else:
#             exit('unknown activation function: %s' % net_act)

#     def _get_pooling(self, net_pooling):
#         if net_pooling == 'maxpooling':
#             return nn.MaxPool2d(kernel_size=2, stride=2)
#         elif net_pooling == 'avgpooling':
#             return nn.AvgPool2d(kernel_size=2, stride=2)
#         elif net_pooling == 'none':
#             return None
#         else:
#             exit('unknown net_pooling: %s' % net_pooling)

#     def _get_normlayer(self, net_norm, shape_feat):
#         if net_norm == 'batchnorm':
#             return nn.BatchNorm2d(
#                 shape_feat[0],
#                 affine=True
#             )

#         elif net_norm == 'layernorm':
#             return nn.LayerNorm(
#                 shape_feat,
#                 elementwise_affine=True
#             )

#         elif net_norm == 'instancenorm':
#             return nn.GroupNorm(
#                 shape_feat[0],
#                 shape_feat[0],
#                 affine=True
#             )

#         elif net_norm == 'groupnorm':
#             return nn.GroupNorm(
#                 4,
#                 shape_feat[0],
#                 affine=True
#             )

#         elif net_norm == 'none':
#             return None

#         else:
#             exit('unknown net_norm: %s' % net_norm)

#     def _make_layers(
#         self,
#         channel,
#         net_width,
#         net_depth,
#         net_norm,
#         net_act,
#         net_pooling,
#         im_size
#     ):
#         layers = []
#         in_channels = channel

#         if im_size[0] == 28:
#             im_size = (32, 32)

#         shape_feat = [
#             in_channels,
#             im_size[0],
#             im_size[1]
#         ]

#         for d in range(net_depth):

#             layers += [
#                 nn.Conv2d(
#                     in_channels,
#                     net_width,
#                     kernel_size=3,
#                     padding=3 if channel == 1 and d == 0 else 1
#                 )
#             ]

#             shape_feat[0] = net_width

#             if net_norm != 'none':
#                 layers += [
#                     self._get_normlayer(
#                         net_norm,
#                         shape_feat
#                     )
#                 ]

#             layers += [
#                 self._get_activation(net_act)
#             ]

#             in_channels = net_width

#             if net_pooling != 'none':
#                 layers += [
#                     self._get_pooling(net_pooling)
#                 ]

#                 shape_feat[1] //= 2
#                 shape_feat[2] //= 2

#         return nn.Sequential(*layers), shape_feat



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
    """AlexNet-style CNN for small images: five conv layers, three 2x2 max-pools and a linear head ``fc``.

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
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, H, W) with H, W in {28, 32}."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        """Return the flattened features before the classifier, shape (N, 192 * 4 * 4)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


''' AlexNetBN '''
class AlexNetBN(nn.Module):
    """AlexNet with a BatchNorm2d layer after every conv (see AlexNet for the architecture).

    Not used by MIPA.
    """
    def __init__(self, channel, num_classes):
        """Create the layers of the network.

        Args:
            channel: Number of input channels (1 or 3).
            num_classes: Number of output logits.
        """
        super(AlexNetBN, self).__init__()
        # first conv uses padding=4 for 1-channel (28x28) inputs, which makes them 32x32; padding=2 otherwise
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel==1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 32x32 input -> three 2x2 max-pools -> a 192-channel 4x4 map, flattened to 192*4*4 features
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, x):
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, H, W) with H, W in {28, 32}."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        """Return the flattened features before the classifier, shape (N, 192 * 4 * 4)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
    """VGG feature extractor with a single linear classifier, for 32x32 inputs.

    Every conv is 3x3 followed by normalization (instance norm as GroupNorm, or batch norm) and ReLU. The
    five 'M' entries of the VGG11 config reduce 32x32 to 1x1, so the classifier sees the 512 channels of
    the last conv. Not used by MIPA.
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
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, 32, 32)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def embed(self, x):
        """Return the flattened last feature map, shape (N, 512) for 32x32 inputs."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
    """VGG11 with batch norm. Not used by MIPA."""
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
        # no global pooling: the flattened last map (4x4 for 32x32 inputs, 3x3 for 1-channel 28x28 inputs since
        # 28 -> 14 -> 7 -> 3) goes straight into the classifier
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

    def embed(self, x):
        """Return the flattened last feature map, shape (N, 512 * expansion * H' * W') (4x4 for 32x32 inputs)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 1x1 average pooling is an identity op; the feature map is flattened as is
        out = F.avg_pool2d(out, kernel_size=1, stride=1) # modification
        out = out.view(out.size(0), -1)
        return out

def ResNet18BN_AP(channel, num_classes):
    """ResNet-18 with average-pool blocks and batch norm. Not used by MIPA."""
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    """ResNet-18 with average-pool blocks and instance norm. Not used by MIPA."""
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs with normalization and a skip connection. Not used by MIPA."""
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

    Not used by MIPA.
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

    The stages have 64/128/256/512 base channels (times ``block.expansion``) with strides 1/2/2/2. A fixed
    4x4 average pooling then reduces the last feature map to 1x1, which is only correct for 32x32 inputs
    (4x4 last map); larger inputs would give a larger flattened feature and break the classifier.
    benchmark_networks.py uses adaptive pooling instead. Not used by MIPA.
    """
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        """Create the layers of the network.

        Args:
            block: BasicBlock or Bottleneck (class, not instance).
            num_blocks: Number of blocks in each of the four stages, e.g. [2, 2, 2, 2].
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
        """Return logits (N, num_classes) for ``x`` of shape (N, channel, 32, 32)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 4x4 average pooling collapses the 4x4 last map of a 32x32 input to 1x1 (-> 512 * expansion features)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        """Return the pooled representation, shape (N, 512 * expansion) for 32x32 inputs."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 4x4 average pooling collapses the 4x4 last map of a 32x32 input to 1x1 (-> 512 * expansion features)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18BN(channel, num_classes):
    """ResNet-18 (BasicBlock, [2, 2, 2, 2]) with batch norm. Not used by MIPA."""
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18(channel, num_classes):
    """ResNet-18 (BasicBlock, [2, 2, 2, 2]) with the default instance norm. Not used by MIPA."""
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes)

def ResNet34(channel, num_classes):
    """ResNet-34 (BasicBlock, [3, 4, 6, 3]) with instance norm. Not used by MIPA."""
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet50(channel, num_classes):
    """ResNet-50 (Bottleneck, [3, 4, 6, 3]) with instance norm. Not used by MIPA."""
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes)

def ResNet101(channel, num_classes):
    """ResNet-101 (Bottleneck, [3, 4, 23, 3]) with instance norm. Not used by MIPA."""
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes)

def ResNet152(channel, num_classes):
    """ResNet-152 (Bottleneck, [3, 8, 36, 3]) with instance norm. Not used by MIPA."""
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes)

