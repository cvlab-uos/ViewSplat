<p align="center">
  <h2 align="center"> ViewSplat <br> View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis <br> from Sparse Views </h2>
  <p align="center">
    <strong>CVLAB-UOS</strong>
  </p>
  <h3 align="center">
    <a href="https://cvlab-uos.github.io/ViewSplat/">Project Page</a> | 
    <a href="https://arxiv.org/abs/2603.25265">Paper</a>
  </h3>
</p>

**ViewSplat** is built upon the foundational work of [SPFSplatV2](https://github.com/ranrhuang/SPFSplatV2).

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a>
    </li>
    <li>
      <a href="#camera-conventions">Camera Conventions</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#running-the-code">Running the Code</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
</ol>
</details>

## Installation

1. Clone ViewSplat.
```bash
git clone git@github.com:cvlab-uos/ViewSplat.git
cd ViewSplat
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n viewsplat python=3.11
conda activate viewsplat

conda install -c conda-forge ninja gcc=11 gxx=11
conda install -c conda-forge mkl=2023.1.0 intel-openmp
conda install -c conda-forge mkl-service mkl_fft mkl_random
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install "diff_gauss_pose @ git+https://github.com/slothfulxtx/diff-gaussian-rasterization.git@pose" --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git@055ab3a2e3e611dff66fa82f632e62a315f3b5e7" --no-build-isolation

pip install --no-cache-dir --no-build-isolation -r requirements.txt
```

## Pre-trained Checkpoints


## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## Running the Code
### Training
1. If using MASt3R-based architecture, download the [MASt3R](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) pretrained model and put it in the `./pretrained_weights` directory.

2. Train with:

```bash


```

### Evaluation
#### Novel View Synthesis and Pose Estimation on SPFSplatV2 (MASt3R-based architecture)
```bash

```

#### Novel View Synthesis and Pose Estimation on SPFSplatV2-L (VGGT-based architecture)
```bash


```

## Camera Conventions
We follow the [pixelSplat](https://github.com/dcharatan/pixelsplat) camera system. The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).
The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Acknowledgements
This project is built upon [SPFSplatV2](https://github.com/ranrhuang/SPFSplatV2) by Ranran Huang. We thank the authors for their excellent work.


## Citation

```

```