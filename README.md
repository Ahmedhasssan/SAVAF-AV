# SAVAF-AV

We present a Sparse Audio-Visual 3D rendering scheme with multi-head acoustic field attention (SAVAF) using minimal Gaussian as visual cues. In SAVAF, we implement an explicit 2D mapped patch-based audio-visual attention to guide the binaural audio generation on learned 3D Gaussian primitives. First, SAVAF, considering the 3D geometry, accounts for listener-source spatial relationships and implements adaptive pose-specific Gaussian filtering to reduce memory utilization and inference time for on-device 3D rendering. Second, we implement the combined audio-visual self-attention for sound source localization and sound orientation-based cross-attention to generate noise-free left and right sound channels for the binaural sound synthesis at the receiver end. Combined with audio-visual rendering backbones, the proposed SAVAF achieves the average performance of 25.9 PSNR, 0.83 SSIM, 0.053 LPIPS, 0.13 ENV, and 1.40 Magnitude distance on the SoTA audio-visual RWAVS dataset.


## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline for AV-NeRF Dataset](#training-with-SAVAF)
- [Training for SoundSpaces](#SoundSpaces-training)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)

## Installation

```bash
# Clone the repository
git clone https://github.com/Ahmedhasssan/SAVAF-AV.git
cd SAVAF-AV

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10.0
- PyTorch >= 2.6.0
- torchvision >= 0.20.0
- numpy
- opencv-python
- PIL

## Dataset Preparation

### Supported Datasets

This project supports the following datasets:
- AV-NeRF
- SoundSpaces and NVS-Replay

### Data Structure

Organize your dataset in the following structure:

```
./release/
├── 1
│   ├── binaural_syn_re.wav
│   ├── feats_train.pkl
│   ├── feats_val.pkl
│   ├── frames
│   │   ├── 00001.png
|   |   ├── ...
│   │   ├── 00616.png
│   ├── source_syn_re.wav
│   ├── transforms_scale_train.json
│   ├── transforms_scale_val.json
│   ├── transforms_train.json
│   └── transforms_val.json
├── ...
├── 13
└── position.json
```

### Training Pipeline for AV-NeRF Dataset

The preprocessing pipeline includes:

1. **Visual Rendering**: Train the 3DGS for visual rendering and save the sparse Gaussians for Audio learning.
2. **Audio Synthesis**: Load the locally stored Gaussians and implement binaural audio synthesis using Multihead Acoustic Field Attention Network

### Usage
Note: Ensure that you provide the correct path to the original dataset.
```bash
**Visual learning and rendering**
export CUDA_VISIBLE_DEVICES=0
# DATA_PATH="/home/ah2288/LP_MipNerF/data/nerf_synthetic/hotdog"
for i in {4..13}; do
    python train.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
        --eval \
        --checkpoint_iteration 30010 \
        --iterations 30010 \
        --checkpoint_path "/home/ah2288/AV-3DGS/output/$i" \
        --start_checkpoint  "/home/ah2288/AV-3DGS/output/$i" 
done
```
```bash
# Simple Way
Bash train.sh
```

```bash
**Audio learning and Synthesis**
export CUDA_VISIBLE_DEVICES=0
# DATA_PATH="/home/ah2288/LP_MipNerF/data/nerf_synthetic/hotdog"
WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) 
for i in {12..12}; do
    # torchrun --nproc_per_node=$WORLD_SIZE --master_port=29505 main_mvsplat.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
    echo "Processing dataset $i with $WORLD_SIZE GPUs"
    python train_av.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
        --eval \
        --start_checkpoint  "/home/ah2288/AV-3DGS/output/$i" \
        --checkpoint_iterations 2000 \
        --iterations 10000 \
        --checkpoint_path "/home/ah2288/AV-3DGS/output/$i" \
done
```
```bash
# Simple Way
Bash train_av.sh
```

### Inference only
Provide the av_checkpoints and use flag --eval_aud
```bash
export CUDA_VISIBLE_DEVICES=0
# DATA_PATH="/home/ah2288/LP_MipNerF/data/nerf_synthetic/hotdog"
WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) 
for i in {12..12}; do
    # torchrun --nproc_per_node=$WORLD_SIZE --master_port=29505 main_mvsplat.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
    echo "Processing dataset $i with $WORLD_SIZE GPUs"
    python train_av.py -s /home/ah2288/AV-3DGS/RWAVS_3DGS_data/release/$i\
        --eval \
        --start_checkpoint  "/home/ah2288/AV-3DGS/output/$i" \
        --checkpoint_iterations 2000 \
        --iterations 10000 \
        --checkpoint_path "/home/ah2288/AV-3DGS/output/$i" \
        --av_checkpoint_path "/home/ah2288/AV-3DGS/output/$i/audio_chkpnt10000.pth" \
        --eval_aud
done
```

## Training with Soundspaces Dataset

### Overview

For SoundSpaces and NVS-Replay, follow a different GitHub link: [SAVAF-SoundSpaces](https://github.com/Ahmedhasssan/SAVAF-SoundSpaces.git)

### Evaluation Metrics

Track the following metrics during feature distillation:

- **3D Generation Accuracy**: PSNR, SSIM and LPIPS Scores
- **Audio Synthesis Quality**: MAG distance, ENV distance, EDT, T60 and C50
- **Audio Synthesis Resource Utilization**: Memory and FPS

## Results

### Performance Comparison

| Methods | Modality |  | Office ↓ |  | House ↓ |  | Apt. ↓ |  | Out. ↓ |  | Overall ↓ |  | Memory (MB) | FPS |
|---------|---------|---|---------|---|---------|---|--------|---|--------|---|-----------|---|-------------|-----|
|  | A | V | MAG | ENV | MAG | ENV | MAG | ENV | MAG | ENV | MAG | ENV |  |  |
| Mono-Mono | ✓ | ✗ | 9.27 | 0.41 | 11.89 | 0.42 | 15.12 | 0.47 | 13.96 | 0.47 | 12.56 | 0.45 | - | - |
| Mono-Energy | ✓ | ✗ | 1.54 | 0.14 | 4.31 | 0.18 | 3.91 | 0.19 | 1.63 | 0.13 | 2.85 | 0.16 | - | - |
| Stereo-Energy | ✓ | ✗ | 1.51 | 0.14 | 4.30 | 0.18 | 3.90 | 0.19 | 1.61 | 0.12 | 2.83 | 0.16 | - | - |
| INRAS | ✓ | ✗ | 1.41 | 0.14 | 3.51 | 0.18 | 3.42 | 0.20 | 1.50 | 0.13 | 2.46 | 0.16 | 1.24 | 180 |
| NAF | ✓ | ✗ | 1.24 | 0.14 | 3.26 | 0.18 | 3.35 | 0.19 | 1.28 | 0.12 | 2.28 | 0.16 | 1.10 | 99 |
| VAM | ✓ | ✓ | 0.98 | 0.14 | 2.10 | 0.16 | 2.33 | 0.20 | 0.89 | 0.12 | 1.57 | 0.16 | 186.8 | 66 |
| AV-NeRF | ✓ | ✓ | 0.93 | 0.13 | 2.01 | 0.16 | 2.23 | 0.18 | 0.85 | 0.11 | 1.50 | 0.15 | 48 | 79 |
| ViGAS | ✓ | ✓ | 0.94 | 0.13 | 2.08 | 0.16 | 2.29 | 0.19 | 0.86 | 0.11 | 1.52 | 0.15 | 52.4 | 34 |
| AV-GS | ✓ | ✓ | 0.86 | 0.12 | 1.97 | 0.15 | 2.03 | 0.18 | 0.79 | 0.11 | 1.42 | 0.14 | 18.40 | 12.5 |
| AV-Cloud | ✓ | ✓ | 0.93 | 0.13 | 2.10 | 0.16 | 2.28 | 0.19 | 0.86 | 0.107 | 1.53 | 0.15 | 15.64 | 83 |
| SAVAF (our) | ✓ | ✓ | **0.85** | **0.12** | **1.90** | **0.14** | **2.08** | **0.17** | **0.80** | **0.10** | **1.40** | **0.13** | **5.44** | **115** |

### Ablation Studies

Results showing the impact of different components:

| Methods | MAG | ENV | Memory (MB) |
|---------|-----|-----|-------------|
| Baseline | 1.50 | 0.150 | 48 |
| **SAVAF** | **1.40** | **0.130** | **5.44** |
| w MLP | 1.45 | 0.140 | 49.5 |
| w/o post-proc. | 0.143 | 0.136 | 4.60 |
| w head-dim 4 | 1.46 | 0.137 | 3.50 |

## Citation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and the open-source community
- Special thanks to [AV-NeRF](https://liangsusan-git.github.io/project/avnerf/) and [NVS](https://arxiv.org/abs/2301.08730), which inspired this work.
- We have borrowed some code from [AV-NeRF](https://github.com/liangsusan-git/AV-NeRF) and [NVS]([https://github.com/facebookresearch/FixRes](https://github.com/facebookresearch/novel-view-acoustic-synthesis)) for dataset loader preparation and baseline.
