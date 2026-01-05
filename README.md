# Teeth Water Removal (Baseline: pix2pix)

This repository contains the baseline experiment for removing water artifacts from intraoral scanner images using Generative Adversarial Networks (GANs).

## 🎯 Project Goal
The goal is to digitally remove water/saliva from dental scans to improve the visibility of tooth structures. The current approach uses the **pix2pix** architecture (paired image-to-image translation).

## 📂 Dataset
The dataset consists of paired images:
- **Input (A):** Wet teeth (with water droplets/reflections).
- **Target (B):** Dry teeth (air-dried references).

### Structure & Split
Total images: ~800 pairs.
- **Training Set:** 700 pairs (Folder: `./wet` and `./dry`)
- **Validation Set:** 100 pairs (Folder: `./val/wet` and `./val/dry`)
- **Resolution:** Images were resized/cropped to 256x256 during training.

## 🚀 Training (Baseline)
We trained a standard pix2pix model on an HPC cluster (A100 GPU).

**Configuration:**
- **Model:** pix2pix (U-Net 256 generator + PatchGAN discriminator)
- **Epochs:** 100 (constant LR) + 100 (decay)
- **Batch Size:** 8
- **Direction:** AtoB (Wet -> Dry)

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
