"""

ResNet models for PACiM.

"""

import torch.nn as nn
from module.pac_module import QuantConv2d, PAConv2d, PALinear, PACT
from main.config import cfg
import torch
import time


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PAConv2d(in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    wbit=cfg.wbit_paconv,
                    xbit=cfg.xbit_paconv,
                    operand=cfg.operand_paconv,
                    dynamic_config=cfg.dynamic_config,
                    threshold=cfg.threshold,
                    mode=cfg.mode_paconv,
                    trim_noise=cfg.trim_noise_paconv,
                    device=cfg.device)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return PAConv2d(in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    wbit=cfg.wbit_paconv,
                    xbit=cfg.xbit_paconv,
                    operand=cfg.operand_paconv,
                    dynamic_config=cfg.dynamic_config,
                    threshold=cfg.threshold,
                    mode=cfg.mode_paconv,
                    trim_noise=cfg.trim_noise_paconv,
                    device=cfg.device)


def affine(in_features, out_features):
    return PALinear(in_features,
                    out_features,
                    bias=True,
                    wbit=cfg.wbit_palinear,
                    xbit=cfg.xbit_palinear,
                    operand=cfg.operand_palinear,
                    mode=cfg.mode_palinear,
                    trim_noise=cfg.trim_noise_palinear,
                    device=cfg.device)


def qconv1(in_planes, out_planes):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes,
                       out_planes,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False,
                       wbit=cfg.wbit_qconv,
                       xbit=cfg.xbit_qconv,
                       mode=cfg.mode_qconv,
                       device=cfg.device)


def qconv2(in_planes, out_planes):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes,
                       out_planes,
                       kernel_size=7,
                       stride=2,
                       padding=3,
                       bias=False,
                       wbit=cfg.wbit_qconv,
                       xbit=cfg.xbit_qconv,
                       mode=cfg.mode_qconv,
                       device=cfg.device)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if cfg.PACT:
            self.relu = PACT()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride
        if cfg.PACT:
            self.relu = PACT()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=cfg.cls_num, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = qconv1(3, 64)
        self.conv2 = qconv2(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = affine(512 * block.expansion, num_classes)
        if cfg.PACT:
            self.relu = PACT()
        else:
            self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, PAConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if cfg.large_model:
            x = self.conv2(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_pacim(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34_pacim(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50_pacim(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101_pacim(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152_pacim(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start = time.time()

    import torch
    model = resnet18_pacim()

    for name, module in model.named_modules():
        print('Layer name: {}, Layer instance: {}'.format(name, module))
    in_feat_num = model.fc.in_features
    model.fc = nn.Linear(in_feat_num, 10)

    model.to(device=device)

    dummy_img = torch.randn(1, 3, 32, 32, device=device)
    output = model(dummy_img)

    end = time.time()
    print('Execution Time: {}s.'.format(end-start))

    print(output.shape)
