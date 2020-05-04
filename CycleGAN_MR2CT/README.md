***Recommendation***

Tensorflow 2 Re-implementation of Unpaired CycleGAN originally proposed by  [Jun-Yan Zhu ](https://people.eecs.berkeley.edu/~junyanz/) *et al.* in [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

# Usage

- Prerequisites
    - Tensorflow 2.0 
    - scikit-image, yaml, tqdm
    - Python 3.6
- Dataset
Have your 2D MR and CT slices saved at ./datasets/MRI and ./datasets/CT folders in ".png" format.

- Example of training

        CUDA_VISIBLE_DEVICES=0 python train.py

- Example of testing

        CUDA_VISIBLE_DEVICES=0 python test.py 
   
