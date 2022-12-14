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
pip install -r requirements.txt
```

### Dataset
The CrowdHuman dataset can be downloaded from http://www.crowdhuman.org/. Orgnize the dataset folder in the following structure:
```shell
data
|-- CrowdHuman
    |-- Images
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
We offer the pre-trained weights on CrowdHuman datasets (Some unavailable model weights are coming soon):

| Method | MR | AP | Weight |
|:---:|:---:|:---:| :---:|
| RetinaNet (SPD) | 60.8% | 84.9% |  |
| RetinaNet (SPD+) | 60.5% | 85.0% |  |
| RetinaNet (VPD) | 56.2% | 86.4% | [retina_fpn_vpd](https://drive.google.com/file/d/10z0ueXMSWcvmzCNKpzK2EwSKTT0YEqV0/view?usp=sharing) |
| ATSS (SPD) | 54.8% | 85.4% |  |
| ATSS (SPD+) | 54.0% | 86.6% |  |
| ATSS (VPD) | 52.1% | 87.1% | [atss_fpn_vpd](https://drive.google.com/file/d/1rVeKp20MkmhQrVYtw3FjDErkgt-8Ja0I/view?usp=sharing) |
| FreeAnchor (SPD) | 51.8% | 84.3% |  |
| FreeAnchor (SPD+) | 51.1% | 84.5% |  |
| FreeAnchor (VPD) | 47.4% | 84.7% | [freeanchor_fpn_vpd](https://drive.google.com/file/d/1WKG6IUcVvPia3fC6a0glSBN05RbzDjG0/view?usp=sharing) |

### Acknowledgement
Our code is heavily based on [Crowddet](https://github.com/Purkialo/CrowdDet) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks for their excellent work!
