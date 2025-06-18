#!/bin/bash

# CSIQ数据集测试
python test.py --dataset CSIQ --name CSIQ_hybrid_test --ckpt ./checkpoints/CSIQ_hybrid/best.pth

# TID2013数据集测试
python test.py --dataset TID2013 --name TID2013_hybrid_test --ckpt ./checkpoints/TID2013_hybrid/best.pth

# KADID数据集测试
python test.py --dataset KADID10K --name KADID10K_hybrid_test --ckpt ./checkpoints/KADID10K_hybrid/best.pth
