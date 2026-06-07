# VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/CVPR-2026-blue" alt="CVPR 2026"></a>
  <a href="https://arxiv.org/abs/2603.07952"><img src="https://img.shields.io/badge/arXiv-2603.07952-b31b1b" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

This is the official PyTorch implementation of:

> **VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer**
>
> *CVPR 2026*

## Highlights

- **Language-Free**: VisualAD removes the text encoder entirely and learns anomaly/normal prototypes purely in the visual feature space.
- **Two Learnable Tokens**: An anomaly token and a normal token are inserted into a frozen ViT, interacting with patch tokens through multi-layer self-attention to encode normality and abnormality.
- **SCA & SAF Modules**: Spatial-Aware Cross-Attention (SCA) injects fine-grained spatial evidence into the tokens; Self-Alignment Function (SAF) recalibrates patch features before anomaly scoring.
- **13 Benchmarks**: State-of-the-art performance across 6 industrial and 7 medical zero-shot anomaly detection benchmarks.
- **Backbone Agnostic**: Adapts seamlessly to CLIP (ViT-L/14@336px) and DINOv2 (ViT-L/14).

## Main Results

### Image-Level ZSAD Performance (AUROC / F1-max / AP)

| Dataset | WinCLIP | APRIL-GAN | AnomalyCLIP | AdaCLIP | **VisualAD (CLIP)** | **VisualAD (DINOv2)** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| MVTec-AD | 90.4 / 92.7 / 95.6 | 86.1 / 90.4 / 93.6 | 91.6 / 92.7 / 96.2 | 92.0 / 92.7 / 96.4 | **92.2 / 93.2 / 96.7** | 90.1 / 92.4 / 94.8 |
| VisA | 75.6 / 78.2 / 78.8 | 77.4 / 78.6 / 80.9 | 81.0 / 80.3 / 84.4 | 79.7 / 79.6 / 83.2 | **84.7 / 82.5 / 87.6** | 83.1 / 81.4 / 86.8 |
| BTAD | 68.2 / 67.8 / 70.9 | 73.7 / 68.7 / 69.9 | 88.7 / 86.0 / 90.6 | 90.0 / 87.2 / 91.5 | **94.9 / 93.9 / 97.0** | 88.2 / 84.7 / 89.7 |
| KSDD2 | 93.5 / 86.4 / 94.2 | 90.4 / 82.9 / 92.0 | 91.9 / 84.5 / 93.4 | 94.9 / 90.3 / 96.2 | **98.0 / 93.9 / 98.3** | 97.7 / 93.1 / 98.1 |
| DAGM | 91.8 / 75.8 / 79.5 | 94.4 / 80.3 / 83.9 | 98.0 / 90.6 / 92.4 | 98.3 / 91.5 / 94.2 | **99.5 / 95.0 / 97.8** | 93.2 / 83.9 / 86.1 |
| DTD-Synthetic | 95.1 / 94.1 / 97.7 | 85.5 / 89.1 / 94.0 | 93.7 / 94.3 / 97.4 | 92.1 / 92.4 / 96.3 | **97.5 / 96.6 / 99.1** | 91.0 / 94.4 / 97.4 |

### Pixel-Level ZSAD Performance (AUROC / F1-max / AP / PRO)

| Dataset | WinCLIP | APRIL-GAN | AnomalyCLIP | AdaCLIP | **VisualAD (CLIP)** | **VisualAD (DINOv2)** |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| MVTec-AD | 82.3 / 24.8 / 18.2 / 62.0 | 87.5 / 42.3 / 39.1 / 43.7 | 91.0 / 38.9 / 34.4 / 81.7 | 88.5 / 43.9 / 41.0 / 47.6 | 90.8 / 43.9 / 41.2 / 87.5 | **91.3 / 47.4 / 45.4 / 88.6** |
| VisA | 73.2 / 9.0 / 5.4 / 51.1 | 93.8 / 32.6 / 26.2 / 86.5 | 95.4 / 27.6 / 20.7 / 86.4 | 95.1 / 33.8 / 29.2 / 71.3 | **95.8 / 34.6 / 28.4 / 91.0** | 95.3 / 35.2 / 29.9 / 88.2 |
| BTAD | 72.7 / 18.5 / 12.9 / 27.3 | 91.3 / 40.1 / 37.7 / 21.0 | 93.0 / 47.1 / 41.5 / 71.0 | 87.7 / 42.3 / 36.6 / 17.1 | 91.1 / **49.8** / **43.1** / **80.4** | **93.4** / 42.6 / 38.7 / 76.7 |
| DTD-Synthetic | 79.5 / 16.1 / 9.8 / 51.5 | 94.9 / 60.4 / 61.0 / 33.8 | 97.5 / 55.8 / 52.5 / 87.9 | 95.1 / 58.4 / 56.1 / 34.3 | **98.1** / 64.3 / 65.5 / **94.8** | 96.7 / **65.8** / **67.7** / 92.4 |

> All backbones use ViT-L/14@336px (CLIP) or ViT-L/14 (DINOv2). Full results including medical benchmarks (OCT17, BrainMRI, Brain_AD, HIS, CVC-ClinicDB, Endo, Kvasir) are available in our paper.

## Getting Started

### 1. Environment

```bash
pip install -r requirements.txt
```

Main dependencies: PyTorch >= 2.0, torchvision, timm, scikit-learn, scipy, tqdm.

### 2. Datasets and Data Preparation

VisualAD is evaluated on 13 zero-shot anomaly detection benchmarks across industrial inspection and medical imaging. The dataset packages used by this project are available below:

| Dataset | Domain | Description | Download |
|:--|:--|:--|:--|
| MVTec-AD | Industrial | Object and texture anomaly detection benchmark with image-level labels and pixel-level masks. | [Google Drive](https://drive.google.com/drive/folders/126ZyuMsgrmwJd_7hlNJQitv_liXfRzJt?usp=sharing) |
| VisA | Industrial | Visual anomaly benchmark for industrial object inspection across multiple product categories. | [Google Drive](https://drive.google.com/drive/folders/1cgpyk8Gfpl_GVnA_l-K8tOV67mKOZdZj?usp=sharing) |
| BTAD | Industrial | BTech industrial anomaly detection benchmark with three product categories. | [Google Drive](https://drive.google.com/drive/folders/1GW9kXgxJM0kC2TX3vWuydaiRkcdg7gjG?usp=sharing) |
| KSDD2 | Industrial | Surface defect detection benchmark for production-line visual inspection. | [Google Drive](https://drive.google.com/drive/folders/1DUQM1DTQxBt4PCV6OWo8QgkP_kGMMxof?usp=sharing) |
| DAGM | Industrial | Synthetic textured-surface defect benchmark with annotated defects. | [Google Drive](https://drive.google.com/drive/folders/1gkjep91EdTgDlEuEVfK-iKVr2mzJi81G?usp=sharing) |
| DTD-Synthetic | Industrial | Synthetic anomaly localization benchmark built from texture images. | [Google Drive](https://drive.google.com/drive/folders/1VZIx-k5a6llIlTi2_lNTSJs3zzyelBlV?usp=sharing) |
| OCT17 | Medical | Retinal OCT scans for retinal disease anomaly/classification evaluation. | [Google Drive](https://drive.google.com/drive/folders/1rhR9_7MmHEF48XIxfl-PulUsjboQEdM5?usp=sharing) |
| BrainMRI | Medical | Brain MRI images for brain tumor or lesion anomaly classification. | [Google Drive](https://drive.google.com/drive/folders/12fYW2KBwjDqtDr-ji6iyGBIRkYzkILmk?usp=sharing) |
| Brain_AD | Medical | Brain MRI benchmark for brain tumor or lesion anomaly classification. | [Google Drive](https://drive.google.com/drive/folders/1cEpuZIIANF06U1wbFf9ky7s_OpFsqukh?usp=sharing) |
| HIS | Medical | Histopathology images for abnormal tissue analysis. | [Google Drive](https://drive.google.com/drive/folders/1pB9OVwk83_qgjvfza4z7CztcdEYX3pwE?usp=sharing) |
| CVC-ClinicDB | Medical | Colonoscopy polyp segmentation benchmark with pixel-level annotations. | [Google Drive](https://drive.google.com/drive/folders/1VT_KELFmYRM7vyo3dk3yQycM2gMPJ-N9?usp=sharing) |
| Endo | Medical | Endoscopic anomaly/lesion benchmark with segmentation annotations. | [Google Drive](https://drive.google.com/drive/folders/1DLfLmrnaXCayYAacIyxzm41x1wByv0x8?usp=sharing) |
| Kvasir | Medical | Gastrointestinal endoscopy benchmark for polyp/anomaly segmentation. | [Google Drive](https://drive.google.com/drive/folders/1-JkKE9d3XuX0QrP8eSATawosUK41UuYA?usp=sharing) |

We adopt the same dataset structure and JSON format as [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP). After downloading a dataset, place it under your local dataset directory and make sure `meta.json` exists in the dataset root. We also provide scripts to generate the required JSON metadata files:

```bash
python generate_dataset_json/mvtec.py
python generate_dataset_json/visa.py
```

### 3. Pre-trained Weights

We provide pre-trained checkpoints with the CLIP (ViT-L/14@336px) backbone:

| Training Set | Checkpoint | Evaluation Set |
|:--|:--|:--|
| VisA | `weight/train_on_visa/CLIP.pth` | MVTec-AD and other cross-dataset benchmarks |
| MVTec-AD | `weight/train_on_mvtec/CLIP.pth` | VisA |

### 4. Quick Start

Run the full cross-dataset training and evaluation pipeline (MVTec-AD <-> VisA) with a single command:

```bash
bash scripts/CLIP.sh
```

Please modify the dataset paths in `scripts/CLIP.sh` before running.

## Citation

If you find this work useful, please consider citing:

```bibtex
@InProceedings{Hou_2026_CVPR,
  author    = {Hou, Yanning and Li, Peiyuan and Liu, Zirui and Wang, Yitong and Ruan, Yanran and Qiu, Jianfeng and Xu, Ke},
  title     = {VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {21346-21356}
}
```

## Acknowledgements

This project builds upon [CLIP](https://github.com/openai/CLIP) and [DINOv2](https://github.com/facebookresearch/dinov2). We thank the authors of [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) and [AdaCLIP](https://github.com/caoyunkang/AdaCLIP) for their open-source implementations.

## Contact

If you have any questions, feel free to reach out:

- **Email**: yanning_hou@nudt.edu.cn
- **WeChat**: HYNing777
