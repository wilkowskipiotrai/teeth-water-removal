cat << 'EOF' > README.md
# 🦷 Teeth Water Removal: Comparative Study

This project aims to remove water and saliva artifacts from intraoral scanner images using Deep Learning. We investigate and compare two different architectural approaches.

## 📚 Approaches

| Method | Architecture | Key Characteristic | Status | Report |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | **pix2pix (GAN)** | Adversarial Training | ✅ Completed | [View Report](./baseline_pix2pix/README.md) |
| **SOTA** | **Restormer** | Transformer | 🚧 Planned | [View Report](./Restormer/README.md) |

## 📂 Repository Structure
* `baseline_pix2pix/` - Code, weights, and logs for the GAN-based approach.
* `Restormer/` - Implementation of the Transformer-based State-of-the-Art model.
EOF