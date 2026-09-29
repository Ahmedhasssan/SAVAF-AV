# SAVAF-AV

We present a Sparse Audio-Visual 3D rendering scheme with multi-head acoustic field attention (SAVAF) using minimal Gaussian as visual cues. In SAVAF, we implement an explicit 2D mapped patch-based audio-visual attention to guide the binaural audio generation on learned 3D Gaussian primitives. First, SAVAF, considering the 3D geometry, accounts for listener-source spatial relationships and implements adaptive pose-specific Gaussian filtering to reduce memory utilization and inference time for on-device 3D rendering. Second, we implement the combined audio-visual self-attention for sound source localization and sound orientation-based cross-attention to generate noise-free left and right sound channels for the binaural sound synthesis at the receiver end. Combined with audio-visual rendering backbones, the proposed SAVAF achieves the average performance of 25.9 PSNR, 0.83 SSIM, 0.053 LPIPS, 0.13 ENV, and 1.40 Magnitude distance on the SoTA audio-visual RWAVS dataset.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start (Full Pipeline)](#quick-start-full-pipeline)
- [Training (Step by Step)](#training-step-by-step)
- [Evaluation](#evaluation)
- [RWAVS Scene Categories](#rwavs-scene-categories)
- [Repository Layout](#repository-layout)
- [SoundSpaces Benchmark](#soundspaces-benchmark)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Recommended hardware

**AMD MI300X** (gfx942) is the recommended system for training and evaluation. The provided Docker image, ROCm stack, and `amd_gsplat` build target this platform. Other AMD ROCm GPUs may work but are not the primary test configuration.

### Docker (recommended, AMD ROCm)

The project ships with a ROCm PyTorch Docker image and a persistent dev container.

```bash
git clone https://github.com/Ahmedhasssan/SAVAF-AV.git
cd SAVAF-AV

# First run: builds the image and starts the container
./docker/start-docker.sh

# Force image rebuild
./docker/start-docker.sh --rebuild

# Drop and recreate the container
./docker/start-docker.sh --fresh
```

Inside the container, the repo is mounted at `/workspace/SAVAF-AV` and data at `/workspace/data`.

That image is for AMD GPUs. It does not run on NVIDIA hardware.

### Docker (NVIDIA CUDA)

`docker/Dockerfile.nvidia` builds the NVIDIA path from a PyTorch CUDA image. Pass `TORCH_CUDA_ARCH_LIST` as the compute capability of the GPU you are compiling for. The image caches the LPIPS weights, so containers run without network access.

```bash
docker build --network=host \
    --build-arg TORCH_CUDA_ARCH_LIST=<compute-capability> \
    -f docker/Dockerfile.nvidia -t savaf-av:cuda .
docker run --rm --gpus all savaf-av:cuda \
    "python -c \"import torch, diff_gaussian_rasterization, simple_knn; print(torch.__version__, torch.cuda.get_device_name(0))\""
```

Mount the whole RWAVS `release/` folder (the loader reads `release/position.json`), plus writable `output/` and `logs/` folders:

```bash
mkdir -p output logs
docker run --rm --gpus all --shm-size 32g \
    -v "$PWD/data/release":/workspace/data/release \
    -v "$PWD/output":/workspace/SAVAF-AV/output \
    -v "$PWD/logs":/workspace/SAVAF-AV/logs \
    -e SCENES="1 12" -e N_GPUS=2 -e AV_RESOLUTION="64 180" \
    savaf-av:cuda "bash scripts/run_full_pipeline.sh"
```

This runs stage 1, stage 2, and the MAG/ENV summary described below. To score the visual model of a scene afterwards:

```bash
docker run --rm --gpus all \
    -v "$PWD/data/release":/workspace/data/release \
    -v "$PWD/output":/workspace/SAVAF-AV/output \
    savaf-av:cuda \
    "python render.py -m output/1 -s /workspace/data/release/1 --iteration 30010 --skip_train --quiet && python metrics.py -m output/1"
```

Stage 1 always uses the first visible GPU. To train several scenes at once, start one container per scene with `--gpus device=<id>`.

### Requirements

- Python >= 3.10
- PyTorch >= 2.6 (ROCm build for AMD GPUs)
- See `docker/Dockerfile` for the full dependency list (`amd_gsplat`, `librosa`, `einops`, `scikit-video`, etc.)

## Dataset Preparation

### Supported Datasets

- **RWAVS** — primary dataset for this repo (real-world audio-visual scenes 1–13)
- **SoundSpaces / NVS-Replay** — see [SoundSpaces Benchmark](#soundspaces-benchmark)

### Download RWAVS

Download and extract the dataset into `data/`:

```bash
mkdir -p data
python -m pip install -U huggingface_hub
hf download susanliang/RWAVS RWAVS_Release.zip \
    --repo-type dataset --local-dir data
cd data && unzip -o RWAVS_Release.zip
```

Expected layout (host path `data/release/`, container path `/workspace/data/release/`):

```
release/
├── 1
│   ├── binaural_syn_re.wav
│   ├── feats_train.pkl
│   ├── feats_val.pkl
│   ├── frames/
│   ├── source_syn_re.wav
│   ├── transforms_scale_train.json
│   ├── transforms_scale_val.json
│   ├── transforms_train.json
│   └── transforms_val.json
├── ...
├── 13
└── position.json
```

## Quick Start (Full Pipeline)

`scripts/run_full_pipeline.sh` runs all three stages end-to-end:

1. **Stage 1** — visual 3DGS (`train.py`) → `output/<scene>/chkpnt30010.pth`
2. **Stage 2** — audio-visual training (`train_av_parallel.sh`) → `output/<scene>/audio_chkpnt*.pth`
3. **Stage 3** — checkpoint sweep + **RWAVS Scene Categories** table (`scripts/eval_checkpoints.sh`)

```bash
cd /workspace/SAVAF-AV

export AV_RESOLUTION="64 180"
# All 13 scenes; stage 2 uses 2 GPUs in parallel
N_GPUS=2 bash scripts/run_full_pipeline.sh

# Office scenes only (1–5)
SCENES="1 2 3 4 5" N_GPUS=2 bash scripts/run_full_pipeline.sh
```

Useful flags:


| Variable        | Default    | Description                                   |
| --------------- | ---------- | --------------------------------------------- |
| `SCENES`        | `1 2 … 13` | Scenes to process                             |
| `N_GPUS`        | `2`        | Parallel GPUs for stage 2                     |
| `STAGE1_GPU`    | `0`        | GPU for sequential stage 1                    |
| `EVAL_GPU`      | `0`        | GPU for stage 3 eval sweep                    |
| `SKIP_STAGE1`   | `0`        | Set `1` if Gaussians already exist            |
| `SKIP_STAGE2`   | `0`        | Set `1` to skip AV training                   |
| `SKIP_EVAL`     | `0`        | Set `1` to skip final summary                 |
| `SKIP_EXISTING` | `0`        | Set `1` to skip scenes with final checkpoints |
| `AV_RESOLUTION` | `64 180`   | Feature-map height/width for stage 2 and eval |


Logs are written to `logs/stage1/scene_<N>.log` and `logs/stage2/scene_<N>.log`.

### Model resolution (`AV_RESOLUTION`)

Stage 2 projects 3D Gaussians onto a 2D feature map before audio attention. The map size, and the parameter memory of the projection layer, is controlled by `AV_RESOLUTION="HEIGHT WIDTH"` (or `--av-resolution HEIGHT WIDTH` in `train_av.py`).


| Setting           | Resolution | Approx. `feature_proj` size | Notes                                   |
| ----------------- | ---------- | --------------------------- | --------------------------------------- |
| Default (compact) | `64 180`   | ~5.8 MB                     | Recommended; within the 5–10 MB target |
| Mid               | `85 240`   | ~10 MB                      | Upper end of the target                 |
| Full              | `170 480`  | ~42 MB                      | Exceeds the recommended memory target   |


```bash
# Compact model — retrain stage 2 after changing the resolution
AV_RESOLUTION="64 180" SCENES="6 12" N_GPUS=2 bash scripts/train_av_parallel.sh

# Eval must use the same resolution as training
AV_RESOLUTION="64 180" bash scripts/eval_checkpoints.sh 6 12
```

Stage 1 visual Gaussians are unchanged; only stage 2 needs to be re-run when you change resolution.

## Training (Step by Step)

### Stage 1: Visual 3D Gaussian Splatting

Trains sparse Gaussians for each scene (~30k iterations per scene).

```bash
cd /workspace/SAVAF-AV
bash scripts/train.sh
```

Or a single scene:

```bash
HIP_VISIBLE_DEVICES=0 python train.py \
    -s /workspace/data/release/1 \
    -m /workspace/SAVAF-AV/output/1 \
    --eval \
    --iterations 30010 \
    --checkpoint_iterations 30010 \
    --checkpoint_path /workspace/SAVAF-AV/output/1
```

### Stage 2: Audio-Visual Training

Loads frozen Gaussians from stage 1 and trains `MixDiffWithCrossAttention` (~10k iterations).

**Single scene:**

```bash
bash scripts/train_av.sh 1          # scene 1 on GPU 0
HIP_VISIBLE_DEVICES=1 bash scripts/train_av.sh 5
```

**Multiple scenes in parallel** (one GPU per scene, auto-queues when GPUs are busy):

```bash
SCENES="1 2 3 4 5" N_GPUS=2 bash scripts/train_av_parallel.sh
SCENES="6 7 8 9 10 11 12 13" N_GPUS=7 bash scripts/train_av_parallel.sh
```

Per-scene logs: `logs/stage2/scene_<N>.log`

## Evaluation

### Checkpoint sweep

`scripts/eval_checkpoints.sh` evaluates every `audio_chkpnt*.pth` for the given scenes, picks the best checkpoint per scene (min ENV, then MAG), and prints results grouped by RWAVS category with comparison to the published SAVAF (our) baseline.

```bash
# Single scene
HIP_VISIBLE_DEVICES=0 bash scripts/eval_checkpoints.sh 6

# Multiple scenes
bash scripts/eval_checkpoints.sh 6 10 12
SCENES="1 2 3 4 5" bash scripts/eval_checkpoints.sh
```

### Inference only

Evaluate a specific audio checkpoint without retraining:

```bash
python train_av.py \
    -s /workspace/data/release/12 \
    -m /workspace/SAVAF-AV/output/12 \
    --eval \
    --start_checkpoint /workspace/SAVAF-AV/output/12 \
    --checkpoint_path /workspace/SAVAF-AV/output/12 \
    --av_checkpoint_path /workspace/SAVAF-AV/output/12/audio_chkpnt10000.pth \
    --av-resolution 64 180 \
    --eval_aud
```

### Visual metrics (PSNR / SSIM / LPIPS)

After stage 1, render the test split and run standard 3DGS metrics (LPIPS uses VGG):

```bash
python render.py -m /workspace/SAVAF-AV/output/12 -s /workspace/data/release/12 --iteration 30010 --skip_train --quiet
python metrics.py -m /workspace/SAVAF-AV/output/12
```

## RWAVS Scene Categories

The RWAVS benchmark groups 13 scenes into four environment types (same mapping as [AV-NeRF](https://github.com/liangsusan-git/AV-NeRF)):


| Category  | Scenes | Metrics  |
| --------- | ------ | -------- |
| Office    | 1–5    | Office ↓ |
| House     | 6–8    | House ↓  |
| Apartment | 9–11   | Apt. ↓   |
| Outdoor   | 12–13  | Out. ↓   |


`scripts/eval_checkpoints.sh` and `scripts/run_full_pipeline.sh` (stage 3) print a **RWAVS Scene Categories** table with MAG and ENV averages per category and overall.

## Repository Layout

```
SAVAF-AV/
├── docker/
│   ├── Dockerfile              # AMD ROCm image
│   ├── Dockerfile.nvidia       # NVIDIA CUDA image
│   ├── docker-entrypoint.sh
│   └── start-docker.sh         # ROCm dev container
├── scripts/
│   ├── train.sh                # Stage 1 launcher (all scenes)
│   ├── train_av.sh             # Stage 2 launcher (single scene)
│   ├── train_av_parallel.sh    # Stage 2 parallel launcher
│   ├── run_full_pipeline.sh    # End-to-end pipeline (stage 1 + 2 + eval)
│   ├── eval_checkpoints.sh     # Checkpoint sweep + MAG/ENV summary
│   └── train_soundspaces_av.sh
├── train.py                    # Stage 1: visual 3DGS
├── train_av.py                 # Stage 2: audio-visual model
├── model.py                    # MixDiffWithCrossAttention and helpers
├── data.py                     # RWAVSDataset loader
├── train_av_ss.py              # SoundSpaces variant (separate benchmark)
├── output/                     # Checkpoints (gitignored)
├── logs/                       # Training logs (gitignored)
└── data/release/               # RWAVS dataset (gitignored)
```

## SoundSpaces Benchmark

For SoundSpaces and NVS-Replay, use the dedicated repo: [SAVAF-SoundSpaces](https://github.com/Ahmedhasssan/SAVAF-SoundSpaces.git)

This repo retains `train_av_ss.py`, `scripts/train_soundspaces_av.sh`, and `datasets/` for SoundSpaces experiments.

### Evaluation Metrics

- **3D Generation Accuracy**: PSNR, SSIM, LPIPS
- **Audio Synthesis Quality**: MAG distance, ENV distance, EDT, T60, C50
- **Resource Utilization**: Memory (MB), FPS

## Citation

```bibtex
@inproceedings{hasssan2026savaf,
  title={SAVAF: Sparse Audio-Visual Rendering with Multihead Acoustic Field Attention Network},
  author={Hasssan, Ahmed and Meng, Jian and Park, Sungjin and Seo, Jae-sun},
  booktitle={International Conference on Pattern Recognition},
  pages={78--93},
  year={2026},
  organization={Springer}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and the open-source community
- Special thanks to [AV-NeRF](https://liangsusan-git.github.io/project/avnerf/) and [NVS](https://arxiv.org/abs/2301.08730), which inspired this work
- Code adapted from [AV-NeRF](https://github.com/liangsusan-git/AV-NeRF) and [NVS](https://github.com/facebookresearch/novel-view-acoustic-synthesis) for dataset loading and baselines

