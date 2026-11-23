"""Residual networks with dataset-specific stems.

Not used by MIPA: no other file imports this module (the ResNets that MIPA uses live in
benchmark_networks.py).

``ResNet`` is a torchvision-style ResNet whose first block (the "stem") is a pluggable module, so the same
four-stage trunk can take inputs of different resolution. The stems below bring the image to the
resolution that stage 1 expects:
    * StemCIFAR:        3x3 stride-1 conv, resolution unchanged (32x32 -> 32x32),
    * StemSTL:          CIFAR stem + 3x3 max-pool with stride 3 (96x96 STL-10 images -> 32x32),
    * StemTinyImageNet: CIFAR stem + 2x2 max-pool (64x64 -> 32x32),
    * StemImageNet:     7x7 stride-2 conv + 3x3 stride-2 max-pool (224x224 -> 56x56).
With ``num_classes=None`` the linear head is omitted and ``forward`` returns the globally pooled
representation of size ``512 * block.expansion`` (also stored as ``representation_dim``).

BasicBlock, Bottleneck and conv1x1 are taken from torchvision. Inputs are float tensors (N, 3, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


class StemCIFAR(nn.Module):
    """Stem for 32x32 inputs: 3x3 stride-1 conv (3 -> 64 channels) + BatchNorm + ReLU, no downsampling."""
    def __init__(self):
        """Create the 3 -> 64 conv and its BatchNorm; ``in_planes`` records the stem's output width (64)."""
        super(StemCIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

    def forward(self, inputs):
        """Map images (N, 3, H, W) to feature maps (N, 64, H, W)."""
        return F.relu(self.bn1(self.conv1(inputs)))

class StemSTL(StemCIFAR):
    """Stem for 96x96 STL-10 images: the CIFAR stem followed by a 3x3 max-pool, giving (N, 64, 32, 32) maps."""
    def __init__(self):
        """Build the CIFAR stem and add the max-pool (its stride defaults to the kernel size, 3)."""
        super(StemSTL, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3)

    def forward(self, inputs):
        """Map images (N, 3, 96, 96) to feature maps (N, 64, 32, 32)."""
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.maxpool(out)
        return out

class StemTinyImageNet(StemCIFAR):
    """Stem for 64x64 Tiny ImageNet images: the CIFAR stem followed by a 2x2 max-pool, giving (N, 64, 32, 32) maps."""
    def __init__(self):
        """Build the CIFAR stem and add the 2x2 max-pool (stride 2)."""
        super(StemTinyImageNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, inputs):
        """Map images (N, 3, 64, 64) to feature maps (N, 64, 32, 32)."""
        out = F.relu(self.bn1(self.conv1(inputs)))
        out = self.maxpool(out)
        return out

class StemImageNet(nn.Module):
    """Standard ImageNet stem: 7x7 stride-2 conv + BatchNorm + ReLU + 3x3 stride-2 max-pool (4x downsampling)."""
    def __init__(self):
        """Create the stem layers.

        Note that the width attribute is called ``inplanes`` here but ``in_planes`` in the CIFAR-style stems.
        """
        super(StemImageNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Map images (N, 3, H, W) to feature maps (N, 64, H/4, W/4), e.g. 224x224 -> 56x56."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
class ResNet(nn.Module):
    """Torchvision-style ResNet trunk with a configurable stem and an optional linear head.

    The stem is followed by four stages of ``block`` with 64/128/256/512 base channels (strides 1, 2, 2, 2),
    then global average pooling to a vector of size 512 * block.expansion and, if ``num_classes`` is given,
    a linear classifier. Adapted from torchvision.models.ResNet by replacing its fixed ImageNet stem with the
    ``stem`` argument. Not used by MIPA.
    """

    def __init__(self, block, layers, num_classes=None, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, stem=StemCIFAR):
        """Create the layers of the network.

        Args:
            block: Residual block class (torchvision BasicBlock or Bottleneck).
            layers: Number of blocks in each of the four stages, e.g. [2, 2, 2, 2] for ResNet-18.
            num_classes: Size of the linear head; None builds no head, so forward() returns the pooled features.
            zero_init_residual: If True, zero-initialize the last norm layer of every residual branch so that
                each block starts out as an identity mapping.
            groups: Number of convolution groups (ResNeXt); 1 for a plain ResNet.
            width_per_group: Base width of the grouped bottleneck convs (Wide ResNet / ResNeXt).
            replace_stride_with_dilation: Three booleans; for stages 2-4, replace the stride-2 downsampling by a
                dilated convolution instead (None means all False).
            norm_layer: Constructor of the normalization layer (default nn.BatchNorm2d).
            stem: Stem class (not an instance), e.g. StemCIFAR; it maps the image to a 64-channel feature map.
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # running input width used by _make_layer; starts at the stem's output width (64)
        self.inplanes = 64
        # current dilation of the 3x3 convs; only grows if replace_stride_with_dilation is used
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # the stem argument is a class; instantiate it (it replaces torchvision's conv1/bn1/relu/maxpool)
        self.stem = stem()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # no classification head when num_classes is None: forward() then returns the pooled features
        if num_classes is not None:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = None

        # He (kaiming) initialization for convs; unit scale and zero shift for norm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # size of the vector returned by forward() when there is no head
        self.representation_dim = 512 * block.expansion

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Build one stage of ``blocks`` residual blocks with ``planes`` base channels.

        Only the first block uses ``stride`` and the projection shortcut (``downsample``); the remaining blocks
        keep the resolution. With ``dilate=True`` the stride is replaced by dilation of the 3x3 convs.
        Updates ``self.inplanes`` so that the next stage sees the right input width.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # dilation instead of striding: keeps the spatial resolution while still growing the receptive field
        if dilate:
            self.dilation *= stride
            stride = 1
        # 1x1 conv + norm shortcut whenever the resolution or the channel count changes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # first block: carries the stride and the downsample shortcut; it uses the dilation of the previous stage
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # remaining blocks: stride 1, identity shortcut, current dilation
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """Run stem -> four stages -> global average pooling -> optional linear head.

        Input is (N, 3, H, W). The output is (N, num_classes) with a head, otherwise the pooled representation
        of shape (N, 512 * block.expansion).
        """
        # See note [TorchScript super()]
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # self.fc is None when no head was requested: return the pooled features unchanged
        if self.fc:
            x = self.fc(x)

        return x

    def forward(self, x):
        """Return the output of ``_forward_impl`` (kept as a separate method for TorchScript, as in torchvision)."""
        return self._forward_impl(x)

def ResNet10(**kwargs):
    """ResNet-10 (BasicBlock, [1, 1, 1, 1]); ``kwargs`` go to ResNet (num_classes, stem, norm_layer, ...). Not used by MIPA."""
    return ResNet(BasicBlock, [1,1,1,1], **kwargs)

def ResNet18(**kwargs):
    """ResNet-18 (BasicBlock, [2, 2, 2, 2]); ``kwargs`` go to ResNet. Not used by MIPA."""
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet34(**kwargs):
    """ResNet-34 (BasicBlock, [3, 4, 6, 3]); ``kwargs`` go to ResNet. Not used by MIPA."""
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    """ResNet-50 (Bottleneck, [3, 4, 6, 3]); ``kwargs`` go to ResNet. Not used by MIPA."""
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    """ResNet-101 (Bottleneck, [3, 4, 23, 3]); ``kwargs`` go to ResNet. Not used by MIPA."""
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    """ResNet-152 (Bottleneck, [3, 8, 36, 3]); ``kwargs`` go to ResNet. Not used by MIPA."""
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)