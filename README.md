# Teeth Water Removal (Baseline: pix2pix)

This repository contains the baseline experiment for removing water artifacts from intraoral scanner images using Generative Adversarial Networks (GANs).

## 🎯 Project Goal
The goal is to digitally remove water/saliva from dental scans to improve the visibility of tooth structures. The current approach uses the **pix2pix** architecture (paired image-to-image translation).

## 📂 Dataset Details
The data was sourced from the internal dataset `_public/Vident-Water`.

### 1. Training Set
- **Source Path:** `_public/Vident-Water/I/1pb_1/camera1`
- **Input (Wet):** 680 images (Range: `frame_00650` to `frame_01329`)
- **Target (Dry):** 1 reference image (`frame_00000`)
- **Note:** The single dry reference image was paired with all 680 wet frames for training.

### 2. Validation / Test Set
- **Source Path:** `_public/Vident-Water/I/1pb_2/camera1/`
- **Input (Wet):** 100 images (Range: `frame_00875` to `frame_00974`)
- **Split:** These images were strictly excluded from the training process to evaluate generalization.

### Preprocessing
- Images were resized and cropped to 256x256 resolution during training to fit the standard pix2pix architecture.

## 🚀 Training Configuration
We trained a standard pix2pix model on an HPC cluster (A100 GPU).

**Hyperparameters:**
- **Model:** pix2pix (U-Net 256 generator + PatchGAN discriminator)
- **Epochs:** 100 (constant LR) + 100 (linear decay)
- **Batch Size:** 8
- **Direction:** AtoB (Wet -> Dry)
- **Optimizer:** Adam (lr=0.0002, beta1=0.5)

**Run Command:**
```bash
python train.py \
  --dataroot . \
  --name teeth_full_run_v1 \
  --model pix2pix \
  --dataset_mode teeth \
  --direction AtoB \
  --use_wandb \
  --wandb_project_name teeth-removal-full \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --batch_size 8 \
  --preprocess resize_and_crop \
  --save_epoch_freq 5 \
  --display_freq 100
