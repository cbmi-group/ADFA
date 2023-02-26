# ADFA

PyTorch implementation of *ADFA: Attention-augmented Differentiable top-k Feature Adaptation for Unsupervised Medical Anomaly Detection*

## Getting Started

Install packages with:

```
$ pip install -r requirements.txt
```

## Dataset 

### Prepare medical image as:

``` 
train data:
    dataset_path/class_name/train/good/any_filename.png
    [...]

test data:
    dataset_path/class_name/test/good/any_filename.png
    [...]

    dataset_path/class_name/test/defect_type/any_filename.png
    [...]
``` 

### Download datasets

BrainMRI : Download from [Kaggle website](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

BUSI : Download from [Kaggle website](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

Covid19 : Download from [Kaggle website](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets)

SipakMed : Download from [Kaggle website](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)

## How to train

### Example
```
python trainer.py --class_name all --data_path [/path/to/dataset/] 
```

## Performance 
 **Datasets** | **$\varepsilon$=0** | **$\varepsilon$=0.01** | **$\varepsilon$=0.05** | **$\varepsilon$=0.1** | **$\varepsilon$=0.2** | **pytorch top-k** | **randomly initialized WR50** 
--------------|---------------------|------------------------|------------------------|-----------------------|-----------------------|-------------------|--------------------------------
 **BrainMRI** | 0.858               | 0.858                  | 0.858                  | 0.857                 | 0.855                 | 0.844             | 0.577                          
 **Covid**    | 0.963               | 0.967                  | 0.967                  | 0.973                 | 0.97                  | 0.917             | 0.47                           
 **BUSI**     | 0.958               | 0.959                  | 0.962                  | 0.966                 | 0.965                 | 0.952             | 0.591                          
 **SIPaKMeD** | 0.964               | 0.965                  | 0.971                  | 0.972                 | 0.972                 | 0.958             | 0.512                          




## Reference
[1] https://github.com/sungwool/CFA_for_anomaly_localization

[2] https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

[4] https://github.com/lukasruff/Deep-SVDD-PyTorch

[4] https://github.com/BangguWu/ECANet

[5] https://github.com/yerkojahve/Meta-Pseudo-label/tree/master/soft_topk

