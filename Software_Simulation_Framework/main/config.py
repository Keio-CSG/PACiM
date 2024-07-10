"""

Parameter management config for training/simulation.

"""

import os

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from easydict import EasyDict

# Obtain value via .key like key-value pair
cfg = EasyDict()

# <|Specify a task|>
# cfg.task = 'CIFAR-10'
cfg.task = 'CIFAR-100'
# cfg.task = 'ImageNet'

# <|Specify a model|>
cfg.model_name = 'ResNet-18'
# cfg.model_name = 'ResNet-34'
# cfg.model_name = 'ResNet-50'
# cfg.model_name = 'ResNet-101'
# cfg.model_name = 'ResNet-152'
# cfg.model_name = 'VGG11'
# cfg.model_name = 'VGG13'
# cfg.model_name = 'VGG16'
# cfg.model_name = 'VGG19'
# cfg.model_name = 'VGG11-BN'
# cfg.model_name = 'VGG13-BN'
# cfg.model_name = 'VGG16-BN'
# cfg.model_name = 'VGG19-BN'

# <|Specify the input image size|>
# CIFAR-10, CIFAR-100
input_size = 28
crop_size = 32
# ImageNet
# input_size = 224
# crop_size = 256

# <|Specify the running device: 'cpu', 'cuda', 'mps', etc.|>
cfg.device = 'cuda'

# <|Specify the classifier dimension|>
cfg.cls_num = 100

# <|Specify the settings for CONV layers|>
cfg.mode_conv = 'Train' # Mode: Train/Inference/Simulation
cfg.wbit_conv = 8 # Weight bit
cfg.xbit_conv = 8 # Activation bit
cfg.trim_noise_conv = 0.0 # Noise intensity for noise-aware training

# <|Specify the settings for LINEAR layers|>
cfg.mode_linear = 'Train' # Mode: Train/Inference/Simulation
cfg.wbit_linear = 8 # Weight bit
cfg.xbit_linear = 8 # Activation bit
cfg.trim_noise_linear = 0.0 # Noise intensity for noise-aware training

# <|Specify the directory for training/validation/test(simulation) dataset|>
# CIFAR-10
# cfg.train_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar10\archive\cifar10\train'
# cfg.valid_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar10\archive\cifar10\test'
# cfg.test_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar10\archive\cifar10\test'
# CIFAR-100
cfg.train_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100\train_images'
cfg.valid_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100\test_images'
cfg.test_dir = r'E:\Machine_Learning\Dataset\Processing\Cifar100\test_images'
# ImageNet
# cfg.train_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train'
# cfg.valid_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\valid'
# cfg.test_dir = r'E:\Machine_Learning\Dataset\Processing\ImageNet\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\valid'

# <|Specify the pretrained model for noise-aware training or test (simulation)|>
cfg.pretrain_model_path = None  # For training, give a None.
# cfg.pretrain_model_path = r'E:\Machine_Learning\Project\Bit-wise_Simulation_Model_Base\main\run\24-07-09_18-45\checkpoint_best.pkl'

# <|Specify the batch size and workers for training|>
cfg.train_bs = 128
cfg.valid_bs = 128
cfg.train_workers = 16

# <|Specify the batch size and workers for test (simulation)|>
cfg.test_bs = 32
cfg.test_workers = 4

# <|Specify the optimizer parameters for training/noise-aware training|>
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1    # gamma (lr decay param) when using MultiStepLR
cfg.log_interval = 10 # Training log interval
cfg.mixup_alpha = 1.0    # Mixup parameter of beta distribution.
cfg.label_smooth_eps = 0.01 # Label smoothing eps parameter.
# Settings for initial training
cfg.lr_init = 0.01
cfg.milestones = [120, 180]
cfg.max_epoch = 200
cfg.mixup = True    # Use mixup or not.
cfg.label_smooth = True # Use label smoothing or not.
# Settings for noise-aware training
# cfg.lr_init = 0.001
# cfg.milestones = [50, 75]
# cfg.max_epoch = 80
# cfg.mixup = False    # Use mixup or not.
# cfg.label_smooth = False # Use label smoothing or not.

#####################################################################################################################
# No need to modify the following params in general
#####################################################################################################################
assert cfg.task in ['CIFAR-10', 'CIFAR-100', 'ImageNet'], "Invalid task specified. Choose from 'CIFAR-10', 'CIFAR-100', or 'ImageNet'."
if cfg.task == 'CIFAR-10':
    norm_mean = [0.4914, 0.4822, 0.4465]  # CIFAR-10
    norm_std = [0.2470, 0.2435, 0.2616]  # CIFAR-10
if cfg.task == 'CIFAR-100':
    norm_mean = [0.5071, 0.4867, 0.4408]    # CIFAR-100
    norm_std = [0.2675, 0.2565, 0.2761]     # CIFAR-100
if cfg.task == 'ImageNet':
    norm_mean = [0.485, 0.456, 0.406]  # ImageNet
    norm_std = [0.229, 0.224, 0.225]  # ImageNet

cfg.transforms_train = transforms.Compose([
    transforms.RandomChoice(
        [
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.3)
        ]
    ),
    transforms.Resize(crop_size),  # Shorter edge = 256
    transforms.CenterCrop(crop_size),
    transforms.RandomCrop(input_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

cfg.transforms_valid = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

cfg.transforms_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

if input_size > 100:
    cfg.large_model = True
else:
    cfg.large_model = False

if __name__ == '__main__':      # Testbench

    from dataset.cifar100 import Cifar100Dataset
    from torch.utils.data import DataLoader
    from tools.common_tools import inverse_transform
    train_data = Cifar100Dataset(root_dir=cfg.train_dir, transform=cfg.transforms_train)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True)

    for epoch in range(cfg.max_epoch):
        for i, data in enumerate(train_loader):

            inputs, labels, dir = data       # B C H W

            img_tensor = inputs[0, ...]     # C H W
            img = inverse_transform(img_tensor, cfg.transforms_train)
            plt.imshow(img)
            plt.show()
            plt.pause(0.5)
            plt.close()
