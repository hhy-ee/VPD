# VPD
This repo contains code of our paper "[Toward Accurate and Robust Pedestrian Detection via Variational Inference]"
## Usage
### Installation
1. Requirements

We have tested the following versions of OS and softwares:

- OS: Ubuntu 18.04.5 LTS
- CUDA: 10.1
- PyTorch 1.6.0
- Python 3.7.13

2. Install all other dependencies via: pip install opencv-python, tqdm

### Dataset
The CrowdHuman dataset can be downloaded from http://www.crowdhuman.org/. The path of the dataset is set in config.py

### Acknowledgement
Our code is heavily based on [Crowddet](https://github.com/Purkialo/CrowdDet) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks for their excellent work!
