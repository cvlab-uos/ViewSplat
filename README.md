<p align="center">
  <h2 align="center"> ViewSplat: View-Adaptive 3D Gaussian Splatting <br> for Feed-Forward Synthesis </h2>
  <h4 align="center"> <b>ECCV 2026</b> </h4>
  <h3 align="center">
    <a href="https://cvlab-uos.github.io/ViewSplat/">Project Page</a> |
    <a href="https://arxiv.org/abs/2603.25265">Paper</a>
  </h3>
</p>
<p align="center">
  <a href="">
    <img src="https://cvlab-uos.github.io/ViewSplat/static/images/intro_figure_v2.png" alt="Overview of ViewSplat" width="90%">
  </a>
</p>
<p align="center">
<strong>ViewSplat</strong> refines Gaussian attributes based on the target-view pose on the fly. <br> This allows for superior reconstruction of fine-grained details.
</p>
<br>

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
Our models are hosted on [Hugging Face](https://huggingface.co/myeon01/ViewSplat)

|                                                    Model name                                              | Training data | Training settings |
|:----------------------------------------------------------------------------------------------------------:|:-------------:|:-------------:|
| [re10k_spf_viewsplat.ckpt](https://huggingface.co/myeon01/ViewSplat/resolve/main/re10k_spf_viewsplat.ckpt) | RE10K | 2 views, SPFSplat-based |
| [acid_spf_viewsplat.ckpt](https://huggingface.co/myeon01/ViewSplat/resolve/main/acid_spf_viewsplat.ckpt) | ACID | 2 views, SPFSplat-based |
| [re10k_spfv2_viewsplat.ckpt](https://huggingface.co/myeon01/ViewSplat/resolve/main/re10k_spfv2_viewsplat.ckpt) | RE10K | 2 views, SPFSplatV2-based |
| [acid_spfv2_viewsplat.ckpt](https://huggingface.co/myeon01/ViewSplat/resolve/main/acid_spfv2_viewsplat.ckpt) | ACID | 2 views, SPFSplatV2-based |
| [re10k_spfv2l_viewsplat.ckpt](https://huggingface.co/myeon01/ViewSplat/resolve/main/re10k_spfv2l_viewsplat.ckpt) | RE10K | 2 views, SPFSplatV2-L-based |
| [acid_spfv2l_viewsplat.ckpt](https://huggingface.co/myeon01/ViewSplat/resolve/main/acid_spfv2l_viewsplat.ckpt) | ACID | 2 views, SPFSplatV2-L-based |

We assume the downloaded weights are located in the `pretrained_weights` directory.

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


### Direct regression ablation

All three encoders (`spf_viewsplat`, `spfv2_viewsplat`, and `spfv2l_viewsplat`)
support `model.encoder.view_dependent_head_type=direct_regression`. The default
is `hypernetwork`, which predicts per-Gaussian view MLPs once and evaluates them
for each target pose.

The direct variant decodes the context tokens and images into a pose-independent
DPT feature map `F` once per encoder call and caches `F` in the encoder output.
For each target pose, it concatenates the per-Gaussian pose input with the cached
`F` and runs only the output convolutions to predict residuals. Training retains
the cache's computation graph so all target losses backpropagate through the same
DPT decode; the cache is scoped to that encoder output.
It predicts Gaussian residuals directly, using the same direction and log-distance
pose input, zero residual initialization, and Gaussian refinement as the hypernetwork.
This preserves the `ViewSplat_direct` architecture and checkpoint shapes while
removing repeated pose-independent DPT work. DPT feature extraction is included
in `encoder` timings; per-target output convolutions and Gaussian updates are
included in `gaussian refining` timings. `vd_mlp_in_dim` specifies the pose input dimension (4) for both modes;
`vd_mlp_hidden_dim` only applies to the hypernetwork.

```bash
# MASt3R: train the direct regression variant
python -m src.main +experiment=spf_viewsplat/re10k \
  model.encoder.view_dependent_head_type=direct_regression \
  wandb.name=re10k_spf_viewsplat_direct_regression

# VGGT: train the direct regression variant
python -m src.main +experiment=spfv2l_viewsplat/re10k \
  model.encoder.view_dependent_head_type=direct_regression \
  wandb.name=re10k_spfv2l_viewsplat_direct_regression

# Evaluate with a checkpoint trained using direct regression
python -m src.main +experiment=spfv2l_viewsplat/re10k mode=test \
  model.encoder.view_dependent_head_type=direct_regression \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
  checkpointing.load=./pretrained_weights/re10k_spfv2l_viewsplat_direct_regression.ckpt
```

The two modes have different view-head weight shapes, so select the matching mode
when loading a ViewSplat checkpoint. Training can use the same base SPFSplat weights
as the corresponding hypernetwork experiment. SPF's `view_dependent_update_*`
switches also apply to direct regression.

## Camera Conventions
We follow the [pixelSplat](https://github.com/dcharatan/pixelsplat) camera system. The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).
The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Acknowledgements
This project is built upon [SPFSplatV2](https://github.com/ranrhuang/SPFSplatV2) by Ranran Huang. We thank the authors for their excellent work.


## Citation

```
@inproceedings{jeong2026viewsplat,
      title={{ViewSplat: View}-Adaptive {3D Gaussian} Splatting for Feed-Forward Synthesis},
      author={Jeong, Moonyeon and Min, Seunggi and Lee, Suhyeon and Seong, Hongje},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2026}
}
```
