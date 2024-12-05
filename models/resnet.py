import logging
import math

from models.layers import *
logger = logging.getLogger(__name__)
model_urls = {
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn_norm(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn_norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = bn_norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GeneralizedMeanPooling2d(nn.Module):
    def __init__(self, p=3.0, output_size=1, eps=1e-6, freeze_p=False):
        super().__init__()
        self.p = p if freeze_p else nn.Parameter(torch.ones(1) * p)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            self.output_size).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'


class SA(nn.Module):
    def __init__(self, conv, kernel_size=5):
        super().__init__()
        self.conv = conv
        num_channels = conv.out_channels
        self.SA = nn.Conv2d(num_channels, num_channels, kernel_size, stride=1, padding=kernel_size // 2,
                            groups=num_channels, bias=False)
        nn.init.dirac_(self.SA.weight, groups=num_channels)

    def forward(self, x):
        return self.SA(self.conv(x))


class ResNet(nn.Module):
    def __init__(self, last_stride, block=Bottleneck, layers=[3, 4, 6, 3], num_class=None):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, nn.BatchNorm2d)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, nn.BatchNorm2d)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, nn.BatchNorm2d)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, nn.BatchNorm2d)

        # head
        self.pooling_layer = GeneralizedMeanPooling2d(p=3.0, freeze_p=True)

        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)

        self.num_class_list = num_class
        self.classifier_list = nn.ModuleList()
        for n in self.num_class_list:
            classifier = nn.Linear(2048, n, bias=False)
            nn.init.normal_(classifier.weight, std=0.001)
            self.classifier_list.append(classifier)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn_norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm))

        return nn.Sequential(*layers)

    def forward(self, x, domain=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feat = self.pooling_layer(x).flatten(1)
        feat = self.bn(feat)
        cls_output = self.classifier_list[domain](feat)

        if self.training:
            return cls_output
        else:
            return feat
