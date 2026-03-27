<p align="center">
  <h2 align="center"> ViewSplat <br> View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis <br>
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
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#running-the-code">Running the Code</a>
    </li>
    <li>
      <a href="#camera-conventions">Camera Conventions</a>
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
Our models are hosted on [Hugging Face]

## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## Running the Code
### Training
1. Download the following pre-trained checkpoints and place them in the `./pretrained_weights` directory:

| Model | Source |
| :---: | :---: |
| **SPFSplat** | [Hugging Face](https://huggingface.co/RanranHuang/SPFSplat/tree/main) |
| **SPFSplatV2** | [Hugging Face](https://huggingface.co/RanranHuang/SPFSplatV2/tree/main) |

```bash
mkdir -p pretrained_weights
# After downloading, your directory should look like this:
# ViewSplat/
# └── pretrained_weights/
#     ├── re10k_spfsplat.ckpt
#     ├── re10k_spfsplatv2l.ckpt
#     └── acid_spfsplatv2l.ckpt
```

2. Train with:

```bash
# 2 view on RealEstate10K, SPFSplat-based architecture (Geometry Transformer: MASt3R)
python -m src.main +experiment=spf_viewsplat/re10k wandb.mode=online wandb.name=re10k_spf_viewsplat

# 2 view on RealEstate10K, SPFSplatV2-L-based architecture (Geometry Transformer: VGGT)
python -m src.main +experiment=spfv2l_viewsplat/re10k wandb.mode=online wandb.name=re10k_spfv2l_viewsplat

# 2 view on ACID, SPFSplatV2-L-based architecture (Geometry Transformer: VGGT)
python -m src.main +experiment=spfv2l_viewsplat/acid wandb.mode=online wandb.name=acid_spfv2l_viewsplat
```

### Evaluation
#### Novel View Synthesis
```bash
# 2 view on RealEstate10K, SPFSplatV2-L-based architecture
python -m src.main +experiment=spfv2l_viewsplat/re10k mode=test wandb.name=re10k_spfv2l_viewsplat_test \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
  checkpointing.load=./pretrained_weights/re10k_spfv2l_viewsplat.ckpt \
  test.save_image=true test.with_offset_only=true

# 2 view on ACID, SPFSplatV2-L-based architecture
python -m src.main +experiment=spfv2l_viewsplat/acid mode=test wandb.name=acid_spfv2l_viewsplat_test \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_acid.json \
  checkpointing.load=./pretrained_weights/acid_spfv2l_viewsplat.ckpt \
  test.save_image=true test.with_offset_only=true
```


## Camera Conventions
We follow the [pixelSplat](https://github.com/dcharatan/pixelsplat) camera system. The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).
The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Acknowledgements
This project is built upon [SPFSplatV2](https://github.com/ranrhuang/SPFSplatV2) by Ranran Huang. We thank the authors for their excellent work.


## Citation

```
@article{Jeong2026viewsplat,
      title={ViewSplat: View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis},
      author={Jeong, Moonyeon and Min, Seunggi and Lee, Suhyeon and Seong, Hongje},
      journal={arXiv preprint arXiv: 2603.25265},
      year={2026}
}
```