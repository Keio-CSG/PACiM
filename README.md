# PACiM
Software Simulation Framework for "PACiM: A Sparsity-Centric Hybrid Compute-in-Memory Architecture via Probabilistic Approximation"

## 1. Overview

* `./Software_Simulation_Framework`: The software simulation framework base for bit-wise CiM simulation.
* `./PACiM`: An software implementation of PACiM in the paper.

You can use the SimConv2d/SimLinear in `./Software_Simulation_Framework/module/module.py` in your own project for bit-wise simulation, and PAConv2d/PALinear in `./PACiM/module/pac_module.py` for PAC simulation.

## 2. Environments

```
python==3.11.7
torch==2.3.0
numpy==1.24.3
matplotlib==3.8.0
easydict==1.13
```

## 3. Software_Simulation_Framework

### 3.1. Framework Overview

This folder includes all modules required for a basic bit-wise CiM simulation.

* `./Software_Simulation_Framework/bins`: Some useful scripts.
* `./Software_Simulation_Framework/dataset`: Datasets for CiM benchmarks.
* `./Software_Simulation_Framework/model`: DNN models for simulations.
* `./Software_Simulation_Framework/tools`: Some useful tools.
* `./Software_Simulation_Framework/module`: CONV/LINEAR modules for QAT/Noise-aware training/Bit-wise simulation.
* `./Software_Simulation_Framework/main`: Main directory for model training and simulation.

All you need to do is to modify the parameter/argument settings `config.py` in the main folder. Then run `src_train.py` for model training and `src_simulation.py` for model evaluation and bit-wise simulation.

When you run `src_train.py`, the training log and model weight will be generated in `main/run`.

### 3.2. Module Introduction

The **SimConv2d** and **SimLinear** module have three different operation mode: **Train**, **Inference**, and **Simulation**. You can specify the mode in the class arguments.

In **Train** mode, the DNN model is trained with fake UINT quantization with given weight/activation bit. If a noise intensity is given with **trim_noise**, the model will be trained under a scaled Gaussian noise (Noise-aware training).

In **Inference** mode, the CONV/LINEAR layers compute output activations with UINT scale. We use this mode to roughly inspect the impact of noise to the DNN inference (by **trim_noise**).

In **Simulation** mode, the module will conduct bit-wise digital CiM simulation. You can modify or overwrite the `_sim_calc()` in the modules to customize and verify your all computing method.

### 3.3. Argument Introduction

We add some arguments to commonly used PyTorch layers and create simulation layers. Below is an example of SimConv2d:

```
SimConv2d(in_planes,                       # Same as nn.conv2d
          out_planes,                      # Same as nn.conv2d
          kernel_size=3,                   # Same as nn.conv2d
          stride=stride,                   # Same as nn.conv2d
          padding=1,                       # Same as nn.conv2d
          bias=False,                      # Same as nn.conv2d
          wbit=cfg.wbit_conv,              # Weight quantization bit-width
          xbit=cfg.xbit_conv,              # Activation quantization bit-width
          mode=cfg.mode_conv,              # Mode selection: Train or Inference or Simulation
          trim_noise=cfg.trim_noise_conv,  # Noise intensity applied to output activations
          device=cfg.device)               # Running device selection: cuda or cpu or mps, etc.
```

## 4. PACiM

### 4.1. Abstract

<a href="https://dl.acm.org/doi/abs/10.1145/3676536.3676704" target="_blank">Paper Link</a>

Approximate computing emerges as a promising approach to enhance the efficiency of compute-in-memory (CiM) systems in deep neural network processing. However, traditional approximate techniques often significantly trade off accuracy for power efficiency, and fail to reduce data transfer between main memory and CiM banks, which dominates power consumption. This paper introduces a novel Probabilistic Approximate Computation (PAC) method that leverages statistical techniques to approximate multiply-and-accumulation (MAC) operations, reducing approximation error by $4\times$ compared to existing approaches. PAC enables efficient sparsity-based computation in CiM systems by simplifying complex MAC vector computations into scalar calculations. Moreover, PAC enables sparsity encoding and eliminates the LSB activations transmission, significantly reducing data reads and writes. This sets PAC apart from traditional approximate computing techniques, minimizing not only computation power but also memory accesses by 50\%, thereby boosting system-level efficiency. We developed PACiM, a sparsity-centric architecture that fully exploits sparsity to reduce bit-serial cycles by 81\% and achieves a peak 8b/8b efficiency of 14.63 TOPS/W in 65 nm CMOS while maintaining high accuracy of 93.85/72.36/66.02\% on CIFAR-10/CIFAR-100/ImageNet benchmarks using a ResNet-18 model, demonstrating the effectiveness of our PAC methodology.

### 4.2. PACiM Overview

The PACiM implementation is based on the Software_Simulation_Framework. We add more arguments to the **SimConv2d** and **SimLinear**, and construct the **PAConv2d** and **PALinear**.

* `./PACiM/mac_noise_experiment`: MAC experiment to evaluate the RMSE of PAC.

### 4.3. Argument Introduction

Upon Software_Simulation_Framework, some arguments for PAC computation are added to the layers. Below is an example of PAConv2d:

```
PAConv2d(in_planes,                         # Same as nn.conv2d
         out_planes,                        # Same as nn.conv2d
         kernel_size=3,                     # Same as nn.conv2d
         stride=stride,                     # Same as nn.conv2d
         padding=1,                         # Same as nn.conv2d
         bias=False,                        # Same as nn.conv2d
         wbit=cfg.wbit_paconv,              # Weight quantization bit-width
         xbit=cfg.xbit_paconv,              # Activation quantization bit-width
         operand=cfg.operand_paconv,        # PAC operand
         dynamic_config=cfg.dynamic_config, # Enable dynamic configuration or not
         threshold=cfg.threshold,           # Threshold for dynamic configuration
         mode=cfg.mode_paconv,              # Mode selection: Train or Inference or Simulation
         trim_noise=cfg.trim_noise_paconv,  # Noise intensity applied to output activations  
         device=cfg.device)                 # Running device selection: cuda or cpu or mps, etc.
```

### 4.4. To reproduce the main results in the paper

We first pretrain the DNN model, then load the pretrained DNN model for noise-aware training. One example of training flow can be summarized as follows: pretrain -> noise-aware training w/ trim_noise = 25.0 -> noise-aware training w/ trim_noise = 50.0 -> noise-aware training w/ trim_noise = 75.0. Tuning the model that can tolerate maximum noise with minimum default accuracy loss. Specific training parameter settings can be found in the `config.py`.

We also provide the model weights of ResNet-18 on CIFAR-100 after noise-aware training for reference: <a href="https://drive.google.com/file/d/1w-izC_dO6da5HAVLZTLXIMJSokkx4wys/view?usp=drive_link" target="_blank">Google Drive</a>

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@inproceedings{zhang2024pacim,
  title={Pacim: A sparsity-centric hybrid compute-in-memory architecture via probabilistic approximation},
  author={Zhang, Wenlun and Ando, Shimpei and Chen, Yung-Chin and Miyagi, Satomi and Takamaeda-Yamazaki, Shinya and Yoshioka, Kentaro},
  booktitle={Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
  pages={1--9},
  year={2024}
}
```
