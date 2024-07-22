"""

VGG models for the simulation framework.

"""

import torch
import torch.nn as nn
from main.config import cfg
from module.module import SimConv2d, SimLinear, PACT


def conv(in_planes, out_planes, stride=1):
    return SimConv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True,
                     wbit=cfg.wbit_conv,
                     xbit=cfg.xbit_conv,
                     mode=cfg.mode_conv,
                     trim_noise=cfg.trim_noise_conv,
                     device=cfg.device)


def affine(in_features, out_features):
    return SimLinear(in_features,
                     out_features,
                     bias=True,
                     wbit=cfg.wbit_linear,
                     xbit=cfg.xbit_linear,
                     mode=cfg.mode_linear,
                     trim_noise=cfg.trim_noise_linear,
                     device=cfg.device)


if cfg.PACT:
    relu = PACT()
else:
    relu = nn.ReLU(inplace=True)


class VGG(nn.Module):
    def __init__(self, features, num_classes=cfg.cls_num, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            affine(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            affine(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            affine(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, SimConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SimLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(str, batch_norm=False):
    layers = []
    if batch_norm:
        layers += [SimConv2d(3,
                             64,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=True,
                             wbit=cfg.wbit_conv,
                             xbit=cfg.xbit_conv,
                             mode=cfg.mode_conv,
                             trim_noise=cfg.trim_noise_conv,
                             device=cfg.device),
                   nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True)]
    else:
        layers += [SimConv2d(3,
                             64,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=True,
                             wbit=cfg.wbit_conv,
                             xbit=cfg.xbit_conv,
                             mode=cfg.mode_conv,
                             trim_noise=cfg.trim_noise_conv,
                             device=cfg.device),
                   nn.ReLU(inplace=True)]

    in_channels = 64
    for v in str:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = conv(in_channels, v)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), relu]
            else:
                layers += [conv2d, relu]
            in_channels = v
    return nn.Sequential(*layers)


if cfg.large_model:

    str = {
        "A": ["M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "B": [64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "D": [64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        "E": [64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    }
else:
    str = {
        "A": [128, "M", 256, 256, 512, 512, 512, 512, "M"],
        "B": [64, 128, 128, "M", 256, 256, 512, 512, 512, 512, "M"],
        "D": [64, 128, 128, "M", 256, 256, 256, 512, 512, 512, 512, 512, 512, "M"],
        "E": [64, 128, 128, "M", 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, "M"]
    }


def vgg11_sim(**kwargs):
    model = VGG(make_layers(str['A']), **kwargs)
    return model


def vgg11_bn_sim(**kwargs):
    model = VGG(make_layers(str['A'], batch_norm=True), **kwargs)
    return model


def vgg13_sim(**kwargs):
    model = VGG(make_layers(str['B']), **kwargs)
    return model


def vgg13_bn_sim(**kwargs):
    model = VGG(make_layers(str['B'], batch_norm=True), **kwargs)
    return model


def vgg16_sim(**kwargs):
    model = VGG(make_layers(str['D']), **kwargs)
    return model


def vgg16_bn_sim(**kwargs):
    model = VGG(make_layers(str['D'], batch_norm=True), **kwargs)
    return model


def vgg19_sim(**kwargs):
    model = VGG(make_layers(str['E']), **kwargs)
    return model


def vgg19_bn_sim(**kwargs):
    model = VGG(make_layers(str['E'], batch_norm=True), **kwargs)
    return model


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = vgg16_bn_sim()
    model.to(device=device)

    for name, module in model.named_modules():
        print('Layer name: {}, Layer instance: {}'.format(name, module))

    # Forward
    fake_img = torch.randn((1, 3, 28, 28), device=device)
    output = model(fake_img)
    print(output.shape)