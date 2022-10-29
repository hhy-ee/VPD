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

2. Install all other dependencies via:
```shell
pip install opencv-python
pip install tqdm
```

### Dataset
The CrowdHuman dataset can be downloaded from http://www.crowdhuman.org/. Orgnize the dataset folder in the following structure:
```shell
<data>
|-- <CrowdHuman>
    |-- <image>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- annotation_train.odgt
    |-- annotation_test.odgt
```

### Train
1. Pretrain weights:

    Download the pretrained model [here](https://github.com/Purkialo/CrowdDet).

2. Config

    Edit config file in `model/<method>/config.py`, including dataset and network settings.

3. Run multi GPU distributed training:
    ```shell
    python tools/train.py -md <method>_fpn_<spd, spd+, vpd>
    ```

### Evaluation
Run the evaluation by:
```shell
python tools/test.py -md <method>_fpn_<spd, spd+, vpd>.py -r <epoch_num>
```
The evaluation results can be found in `model/<method>/outputs/eval_dump` with epoch IDs (epoch_num, ..., 30).

## Result
We offer the pre-trained weights on ResNet-50 FPN:
| Method | MR | AP | Weight |
|:---:|:---:|:---:|:---:| :---:|
| RetinaNet (SPD) | 54.1% | 54.4% |  |
| RetinaNet (SPD+) | 56.0% | 56.3% |  |
| RetinaNet (VPD) | 56.8% | 56.9% |  |

### Acknowledgement
Our code is heavily based on [Crowddet](https://github.com/Purkialo/CrowdDet) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks for their excellent work!
