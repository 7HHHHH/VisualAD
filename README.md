# VisualAD

Official implementation of **VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer**.

## Requirements

```bash
pip install torch torchvision numpy scipy scikit-learn scikit-image tabulate tqdm pyyaml
```

## Training

```bash
python train.py \
    --train_data_path /path/to/dataset \
    --save_path ./checkpoints \
    --train_dataset mvtec \
    --backbone "ViT-L/14@336px" \
    --epoch 15 \
    --batch_size 8 \
    --device cuda:0
```

## Testing

```bash
python test.py \
    --test_data_path /path/to/dataset \
    --checkpoint_path ./checkpoints/epoch_15.pth \
    --test_dataset mvtec \
    --save_path ./results \
    --device cuda:0
```

## Cross-Dataset Evaluation

Run the provided script for cross-dataset experiments (MVTec ↔ VisA):

```bash
bash scripts/final_clip.sh
```
