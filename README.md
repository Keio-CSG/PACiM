# PACiM
Software Simulation Framework for "PACiM: A Sparsity-Centric Hybrid Compute-in-Memory Architecture via Probabilistic Approximation"

## 1. Overview

## 2. Environments

```
python==3.11.7
torch==2.3.0
numpy==1.24.3
matplotlib==3.8.0
easydict==1.13
```

## 3. Under Construction

## 4. PACiM

### Abstract

Approximate computing emerges as a promising approach to enhance the efficiency of compute-in-memory (CiM) systems in deep neural network processing. However, traditional approximate techniques often significantly trade off accuracy for power efficiency, and fail to reduce data transfer between main memory and CiM banks, which dominates power consumption. This paper introduces a novel Probabilistic Approximate Computation (PAC) method that leverages statistical techniques to approximate multiply-and-accumulation (MAC) operations, reducing approximation error by $4\times$ compared to existing approaches. PAC enables efficient sparsity-based computation in CiM systems by simplifying complex MAC vector computations into scalar calculations. Moreover, PAC enables sparsity encoding and eliminates the LSB activations transmission, significantly reducing data reads and writes. This sets PAC apart from traditional approximate computing techniques, minimizing not only computation power but also memory accesses by 50\%, thereby boosting system-level efficiency. We developed PACiM, a sparsity-centric architecture that fully exploits sparsity to reduce bit-serial cycles by 81\% and achieves a peak 8b/8b efficiency of 14.63 TOPS/W in 65 nm CMOS while maintaining high accuracy of 93.85/72.36/66.02\% on CIFAR-10/CIFAR-100/ImageNet benchmarks using a ResNet-18 model, demonstrating the effectiveness of our PAC methodology.

## Citation

If you find this repo is useful, please cite our paper. Thanks.

```bibtex
@inproceedings{PACiM,
  title={PACiM: A Sparsity-Centric Hybrid Compute-in-Memory Architecture via Probabilistic Approximation},
  author={Zhang, Wenlun and Ando, Shimpei and Chen, Yung-Chin and Miyagi, Satomi and Takamaeda-Yamazaki, Shinya and Yoshioka, Kentaro},
  booktitle = {Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
  year={2024},
  doi = {10.1145/3676536.3676704}
}
