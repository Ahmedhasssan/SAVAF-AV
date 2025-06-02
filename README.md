# SAVAF-AV

We present a Sparse Audio-Visual 3D rendering scheme with multi-head acoustic field attention (SAVAF) using minimal Gaussian as visual cues. In SAVAF, we implement an explicit 2D mapped patch-based audio-visual attention to guide the binaural audio generation on learned 3D Gaussian primitives. First, SAVAF, considering the 3D geometry, accounts for listener-source spatial relationships and implements adaptive pose-specific Gaussian filtering to reduce memory utilization and inference time for on-device 3D rendering. Second, we implement the combined audio-visual self-attention for sound source localization and sound orientation-based cross-attention to generate noise-free left and right sound channels for the binaural sound synthesis at the receiver end. Combined with audio-visual rendering backbones, the proposed SAVAF achieves the average performance of 25.9 PSNR, 0.83 SSIM, 0.053 LPIPS, 0.13 ENV, and 1.40 Magnitude distance on the SoTA audio-visual RWAVS dataset.


## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training with Patch Masked Data](#training-with-patch-masked-data)
- [Knowledge Distillation](#knowledge-distillation)
- [Feature Distillation with Patch Masked Data](#feature-distillation-with-patch-masked-data)
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
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    ├── class1/
    └── class2/
```

### Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Patch Masking Preparation**: Generate mask patterns for the train and val datasets. Verify the generated images with the proper unmarked region of interest only.
2. **Normalization**: Apply ImageNet statistics normalization
3. **Data Augmentation**: Random horizontal flip, rotation, and color jittering

### Usage
Note: Make sure to provide the correct path of the original Imagenet-1k data at line 302 of "main_pretrain.py" and pre-trained HPM model at line 45 of "src/data/load.py".
```bash
# Prepare dataset
export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --master_port=29502 --nproc_per_node=1 --nnodes 1 \
    main_pretrain.py \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_base_patch16_dec512d8b \
    --input_size 224 \
    --token_size 14 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path "/scratch/dataset/imagenet-1k" \
    --output_dir  ./output \
    --log_dir   ./log_dir/pretrain \
    --experiment hpm_masked_unmasked_KND_ep800 \
    --learning_loss \
    --relative \
    --eval
```
```bash
# Simple Way
Bash scripts/masked_data_generation.sh
```

### Configuration

Edit the configuration file `config/data_config.yaml`:

```yaml
data:
  output_dir: /scratch/dataset/imagenet_masked/val
  input_size: 224
  color_jitter: 0.4
  aa: 'rand-m9-mstd0.5-inc1'
  train_interpolation: 'bicubic'
  reprob: 0.25
  remode: 'pixel'
  recount: 1

pretrained: "/scratch/checkpoint/in-sensor-computing/output/mae_vit_base_patch16_dec512d8b_hpm_masked_unmasked_KND_ep800_temp.pth"

model:
  norm_pix_loss: False
  vis_mask_ratio: 0.75
```

## Training with Patch Masked Data

### Overview

This section describes the training process using patch masked data, which involves:
- Masking redundant patches that do not cover the desired object
- Training the model to reconstruct or classify from partial information
- Improving model robustness and feature learning

### Training Configuration

Key training parameters:

```yaml
data:
  output_dir: /scratch/dataset/imagenet_masked/val
  input_size: 224
  color_jitter: None
  aa: 'rand-m9-mstd0.5-inc1'
  train_interpolation: 'bicubic'
  reprob: 0.25
  remode: 'pixel'
  recount: 1

pretrained: "/home/ah2288/A-ViT/results/avit_base_patch16_224/checkpoint.pth"

model:
  act_mode: 4
  gate_scale: 10.0
  gate_center: 30
  distr_prior_alpha: 0.001
```

### Training Script

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python3 -m torch.distributed.launch --master_port=29505 --nproc_per_node=6 --nnodes 1 \
    main_finetune.py \
    --batch_size 256 \
    --accum_iter 1 \
    --model vit_base_patch16 \
    --finetune "/scratch/checkpoint/in-sensor-computing/output/mae_vit_base_patch16_dec512d8b_hpm_masked_unmasked_KND_ep800_temp.pth" \
    --epochs 200 \
    --start_epoch 0 \
    --warmup_epochs 5 \
    --blr 5e-4 --layer_decay 0.8 --weight_decay 0.05 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path "/scratch/dataset/imagenet_masked" \
    --nb_classes 1000 \
    --output_dir  ./output \
    --log_dir   ./log_dir/finetune_masked \
    --experiment masked_funetuning \
    --not_resume True
    # --eval
```

### Simple Training Command
```bash
1. Update the finetune_base.sh with the new model checkpoints path and the imagenet-1k data path
2. Bash scripts/finetune_base.sh
```

## Feature Distillation for Training

### Overview

Feature distillation transfers knowledge from a larger teacher model (CLIP, in our case) to a smaller student model (ViT) using patch-masked data. This approach combines the benefits of model compression with robust feature learning.

### Teacher-Student Architecture

```
Teacher Model (Large)    →    Student Model (Compact)
     ↓                             ↓
Feature Maps                 Feature Maps
     ↓                             ↓
Knowledge Transfer Loss    +    Task Loss
```

### Configuration

```yaml
MODEL:
  TYPE: vit
  NAME: fd_pretrain
  DROP_PATH_RATE: 0.1
  VIT:
    EMBED_DIM: 768
    DEPTH: 12
    NUM_HEADS: 12
    USE_APE: False
    USE_RPB: False
    USE_SHARED_RPB: True
    USE_MEAN_POOLING: False
    WITH_CLS_TOKEN: True
DATA:
  IMG_SIZE: 224
  BATCH_SIZE: 128
TRAIN:
  EPOCHS: 100
  WARMUP_EPOCHS: 10
  BASE_LR: 3e-4
  WARMUP_LR: 5e-7
  MIN_LR: 5e-6
  WEIGHT_DECAY: 0.05
  CLIP_GRAD: 3.0
PRINT_FREQ: 100
SAVE_FREQ: 5
TAG: fd_pretrain_clip_vit_base__img224__100ep

```
### Distillation Strategies

- **Feature-based Distillation**: Match intermediate feature maps
- **Distinct-features-based Distillation**: The Teacher Model gets the Original data, and the  Student Model gets the patch-masked data
- **Patch-aware Distillation**: Focus on unmasked regions

### Training Process of Feature-based Distillation
**Use Feature Distillation branch code**
1. **Teacher Preparation**: Load pre-trained teacher model and use masked data
2. **Student Training**: Train the student model with distillation loss and use masked data
3. **Loss Combination**: Combine knowledge distillation and task-specific losses

```bash
# Features distillation training
export CUDA_VISIBLE_DEVICES=1,3,5,6,4,7
python3 -m torch.distributed.launch --master_port=29502 --nproc_per_node=6 --nnodes 1 \
    main_fd.py \
    --cfg "./configs/pretrain/fd_pretrain__clip_vit_base__img224__300ep.yaml"\
    --batch-size 256 \
    --data-path "/scratch/dataset/imagenet_masked/" \
    --output  ./output \
    --enable-amp \
    --use-checkpoint \
    --dual-data False \
    # --eval
```

### Simple Training Command
```bash
1. Update the feature_distillation.sh with the new model checkpoints path and the imagenet-1k data path
2. Bash feature_distillation.sh
```


### Training Process of Distinct-feature-based Distillation
**Use In-sensor-computing branch code**
1. **Teacher Preparation**: Load pre-trained teacher model and use original data
2. **Student Training**: Train the student model with distillation loss and use masked data
3. **Loss Combination**: Combine knowledge distillation and task-specific losses

```bash
# Features distillation training
export CUDA_VISIBLE_DEVICES=1,3,5,6
python3 -m torch.distributed.launch --master_port=29502 --nproc_per_node=1 --nnodes 1 \
    main_pretrain.py \
    --batch_size 128 \
    --accum_iter 1 \
    --model mae_vit_base_patch16_dec512d8b \
    --input_size 224 \
    --token_size 14 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path "/scratch/dataset" \
    --output_dir  ./output \
    --log_dir   ./log_dir/pretrain \
    --experiment hpm_masked_unmasked_KND_ep800 \
    --learning_loss \
    --relative \
    --learn_feature_loss 'dino' \
    --dino_path '/home/ah2288/HPM/dino_vitbase16_pretrain_full_checkpoint.pth' \
    --dual_data
    # --eval
```

### Simple Training Command
```bash
1. Update the pretrain_base.sh with the new model checkpoints path and the imagenet-1k data path
2. Bash scripts/pretrain_base.sh
```

### Benefits with Patch Masking

- Enhanced robustness of compressed models
- Better generalization on partial information
- Improved feature alignment between teacher and student
- Reduced overfitting in student models

### Evaluation Metrics

Track the following metrics during feature distillation:

- **Classification Accuracy**: Task performance on test set
- **Compute Reduction**: FLOPs reduction

### Hyperparameter Tuning

Key hyperparameters to optimize:

```yaml
hyperparameters:
  feature_loss_weight: [0.1, 0.5, 1.0]
  temperature: [3.0, 4.0, 5.0]
  mask_ratio: [0.5, 0.75, 0.85]
  adaptation_layer_dim: [128, 256, 512]
```

## Results

### Performance Comparison

| Masking Ratio | Masking Method | Dataset | Training Method | Epochs | Accuracy |
|---------------|----------------|---------|-----------------|--------|----------|
| 0% | HPM-Baseline | Imagenet-1k | VIT Fine-tuning | 100 | 83.1% |
| 80 % | A-VIT | Imagenet-1k | VIT Fine-tuning | 100 | 64.56% |
| 60 % | A-VIT | Imagenet-1k | VIT Fine-tuning | 100 | 72% |
| 50-60 % | HPM | Imagenet-1k | VIT Fine-tuning | 100 | 77.6% |
| 50-60 % | HPM | Imagenet-1k | ResNet-101, Efficient-Net | 100 | 64.4% |
| 40-50 % | HPM | Imagenet-1k | Feature Distillation (Setting 1) | 100 | 79.82 % |
| 40-50 % | HPM | Imagenet-1k | Feature Distillation (Setting 2) | 100 | 76.92 % |
| 40-50 % | HPM | Imagenet-1k | Feature Distillation (Setting 1) | 300 | 80.82 % |

### Ablation Studies

Results showing the impact of different components:

1. MAE training
2. CNN architecture training (To repeat this experiment, use the cnn_training branch)
3. Feature distillation-based training

## Citation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and the open-source community
- Special thanks to [HPM](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Hard_Patches_Mining_for_Masked_Image_Modeling_CVPR_2023_paper.html), that inspired this work.
- We have borrowed a lot of code from [HPM](https://github.com/Haochen-Wang409/HPM) and [FixRes](https://github.com/facebookresearch/FixRes) for CNN based trainig.
