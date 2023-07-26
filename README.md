# RT-SRTS
This is the code repository for RT-SRTS.

The Reconstruction Toolkit (RTK) is an open source cross-platform software for rapid cone beam CT reconstruction based on the Insight Toolkit (ITK). RTK is an open source package of CBCT reconstruction algorithms, owned by Kitware, and is based on the ITK package extension. RTK implements many existing CT image reconstruction algorithms, including ADMM, SART, SIRT, POCS, etc.

We use RTK to obtain DRR images from CT images at a certain angle.

## Introduction
Our network is structured as follows.
For more details, please read the paper.
![image](network.png)

The AEC module is shown below.
![image1](AEC.png)

The URE module is shown below.
![image2](URE.png)


## Install
```sh
$ git clone https://github.com/ZywooSimple/RT-SRTS.git
$ cd RT-SRTS
$ conda env create -f PatReconSeg.yaml
$ conda activate PatReconSeg
```

## Data
We saved each real 3DCT and corresponding real tumor label into the h5py file and put it into a separate folder according to the 1080 time phase. The structure is as follows:
```sh
|--h5py
|  |--1
|   |--ct_xray12_label.h5
|  |--2
|    |--ct_xray12_label.h5
   ·········
|--ct_rtk
|  |--1_rtk.mha
|  |--2_rtk.mha
|  |--3_rtk.mha
   ·········
```



## Train & Test
To train our model, run:
```python 
$ python RT-SRTS/train.py
```

To test our model, run :
```python 
$ python RT-SRTS/test.py
```



