# MQAF

Paper：Image Quality Assessment: Exploring Quality Awareness via Memory-driven Distortion Patterns Matching

## Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA 10.2+

## Training

```bash
# 混合模式
python train.py --dataset CSIQ --train_mode hybrid --n_epoch 200 --learning_rate 1e-4 --name CSIQ_hybrid
# 纯记忆模式
python train.py --dataset CSIQ --train_mode memory_only --n_epoch 200 --learning_rate 1e-4 --name CSIQ_memory_only

```

## Testing

```bash
python test.py --dataset CSIQ --name CSIQ_hybrid_test --ckpt ./checkpoints/CSIQ_hybrid/best.pth
```
