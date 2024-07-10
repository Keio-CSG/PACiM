"""

Main simulation code for PACiM.

"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from main.config import cfg
from model.resnet_pacim import resnet18_pacim, resnet34_pacim, resnet50_pacim, resnet101_pacim, resnet152_pacim
from model.vgg_pacim import vgg11_pacim, vgg13_pacim, vgg16_pacim, vgg19_pacim, vgg11_bn_pacim, vgg13_bn_pacim, vgg16_bn_pacim, vgg19_bn_pacim
from dataset.cifar10 import Cifar10Dataset
from dataset.cifar100 import Cifar100Dataset
from dataset.imagenet import ImageNetDataset
from tools.common_tools import setup_seed

if __name__ == '__main__': # Main script for simulation

    # Step 1: Dataset
    # Construct Dataset instance, and then construct DataLoader
    if cfg.task == 'CIFAR-10':
        test_data = Cifar10Dataset(root_dir=cfg.test_dir, transform=cfg.transforms_test)
    if cfg.task == 'CIFAR-100':
        test_data = Cifar100Dataset(root_dir=cfg.test_dir, transform=cfg.transforms_test)
    if cfg.task == 'ImageNet':
        test_data = ImageNetDataset(root_dir=cfg.test_dir, transform=cfg.transforms_test)

    test_loader = DataLoader(dataset=test_data, batch_size=cfg.test_bs, shuffle=False, num_workers=cfg.test_workers)

    # Step 2: Model Selection
    model_dic = {
        'ResNet-18': resnet18_pacim(),
        'ResNet-34': resnet34_pacim(),
        'ResNet-50': resnet50_pacim(),
        'ResNet-101': resnet101_pacim(),
        'ResNet-152': resnet152_pacim(),
        'VGG11': vgg11_pacim(),
        'VGG13': vgg13_pacim(),
        'VGG16': vgg16_pacim(),
        'VGG19': vgg19_pacim(),
        'VGG11-BN': vgg11_bn_pacim(),
        'VGG13-BN': vgg13_bn_pacim(),
        'VGG16-BN': vgg16_bn_pacim(),
        'VGG19-BN': vgg19_bn_pacim()
    }
    model = model_dic[cfg.model_name]
    pretrained_state_dict = torch.load(cfg.pretrain_model_path, map_location='cpu')
    model.load_state_dict(pretrained_state_dict['model_state_dict'])
    model.to(cfg.device)
    model.eval()

    # Step 3: Inference (Simulation)
    class_num = cfg.cls_num
    conf_mat = np.zeros((class_num, class_num))

    for i, data in enumerate(test_loader):
        inputs, labels, path_img = data
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

        outputs = model(inputs)

        # Generate confusion matrix
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.
            # Simulation print
            print('Label: {}, Predict: {}.'.format(cate_i, pre_i))

    acc_avg = conf_mat.trace() / conf_mat.sum()
    print('Test Acc: {:.2%}'.format(acc_avg))
