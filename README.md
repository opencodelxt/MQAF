# MQAF

Paper：Image Quality Assessment: Exploring Quality Awareness via Memory-driven Distortion Patterns Matching

## Requirements

- Python 3.8+
- PyTorch 1.7+
- CUDA 10.2+


## Testing
Please download model weights from [[Baidu](https://pan.baidu.com/s/1Eps3Bn0S8B-t5V9Xjn4k8g), Password:12sx] and run

```bash
python test.py --dataset TID2013 --name TID2013_hybrid_test --ckpt PATH_TO_CHECKPOINT_TID2013
```
