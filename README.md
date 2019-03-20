# Learning Transmission Filtering Network for Image-Based PM2.5 Estimation
This repository is for TFN introduced in the following paper

[Yinghong Liao](https://github.com/nicholasly/), Bin Qiu, Zhuo Su, Ruomei Wang, Xiangjian He, "Learning Transmission Filtering Network for Image-Based PM2.5 Estimation", ICME 2019

The code is built on [RESCAN](https://github.com/XiaLiPKU/RESCAN) and [TernausNet](https://github.com/thstkdgus35/EDSR-PyTorch), which is tested on Ubuntu 14.04 environment (Python >= 3.6, PyTorch_0.4.1, CUDA_8.0.61, cuDNN_5.1) with a NVIDIA GeForce GTX Titan X GPU.

If you have any question, please send an email to <nicholasliao23@gmail.com> and we are willing to answer.

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Real Image Test](#Real Image Test)
5. [Results](#results)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## Introduction
PM2.5 is an important indicator of the severity of air pollution and its level can be predicted through hazy photographs caused by its degradation. Image-based PM2.5 estimation is thus extensively employed in various multimedia applications but is challenging because of its ill-posed property. In this paper, we convert it to the problem of estimating the PM2.5-relevant haze transmission and propose a learning model called the transmission filtering network. Different from most methods that generate a transmission map directly from a hazy image, our model takes the coarse transmission map derived from the dark channel prior as the input.  To obtain a transmission map that satisfies the local smoothness constraint without regional boundary degradation, our model performs the edge-preserving smoothing filtering as the refinement on the map. Moreover, we introduce the attention mechanism to the network architecture for more efficient feature extraction and smoothing effects in the transmission estimation. Experimental results prove that our model performs favorably against the state-of-the-art dehazing methods in a variety of hazy scenes.

![CA](/figure/TFN.PNG)
Transmission Filtering Network (TFN) architecture.
![RAB](/figure/RAB.PNG)
Residual Attention Block (RAB) architecture.

## Train
### Prepare training data 

1. Download RESIDE training data from [RESIDE-Standard](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0) and [RESIDE-β](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2).

2. Plaze the hazy images in the folder <code>./dataset/OTS_BETA/train/hazy/</code> and the haze-free images in the folder <code>./dataset/OTS_BETA/train/clear/</code>.

### Begin to train

1. (optional) Download models for our paper and place them in the folder <code>./models/</code>.

    All the trained models can be downloaded from [BaiduYun]().

2. Cd to <code>./config</code>, run the following scripts to train models. We adopt the initialization (warm-up) in network training, if you would like to train the network at the beginning, we recommend to use the model <code>step_10000</code>'. You can set any parameter in the file <code>./config/settings.py</code>.

    ```bash
    # (example) train the network with the model step_10000 
    python main.py -a train -m ../models/step_10000

    ```

3. The corresponding training inputs, targets, results can be found in the path <code>./logdir/</code>.

## Test
### Quick start
1. Download models for our paper and place them in <code>./models/</code>.

    All the trained models can be downloaded from [BaiduYun]().

2. Place the folders <code>./dataset/OTS_BETA/test/indoor/hazy</code> and <code>./dataset/OTS_BETA/test/indoor/clear</code> in the path <code>./dataset/OTS_BETA/test/hazy</code> and <code>./dataset/OTS_BETA/test/clear</code> if you want to test the testing set Indoor. The similar opearions are conducted if you want to test the testing set Outdoor.

3. Cd to <code>./config</code>, run the following scripts to test the testing dataset in batch. The batch size needs to be set as 1 in the file <code>./config/settings.py</code>.

    ```bash
    # (example) test with the model step_100000 
    python main.py -a test -m ../models/step_100000

    ```

4. The corresponding testing inputs, targets and results can be found in the path <code>./logdir/</code>.

## Real Image Test
### Quick start
1. Download models for our paper and place them in <code>./models/</code>.

    All the trained models can be downloaded from [BaiduYun]().

2. Choose a real hazy image from the folder <code>./dataset/OTS_BETA/real/</code>.

3. Cd to <code>./config</code>, run the following scripts to obtain a dehazed result.

    ```bash
    # (example) test a real hazy image with filename as 'tiananmen.bmp' with the model step_100000 
    python main.py -a real -m ../models/step_100000 -n tiananmen.bmp

    ```

4. The corresponding dehazed result can be found in the folder <code>./dataset/OTS_BETA/real_output/</code>.

## Results
### Quantitative Results
![numeric](/figure/numeric.PNG)

Quantitative evaluation with some state-of-the-art methods on the synthetic datasets (SSIM/PSNR), where the best
and the second best numeric values are marked in red and blue, respectively.

For more results, please refer to our [main papar]().
### Visual Results
![visual](/figure/visual.PNG)
Visual comparison with some state-of-the-art dehazing methods on real hazy images.

## Citation
```
@inproceedings{liao_2019_tfn,
    title = {Learning Transmission Filtering Network for Image-Based PM2.5 Estimation},
    author = {Liao, Yinghong and Qiu, Bin and Su, Zhuo and Wang, Ruomei and He, Xiangjian},
    booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
    year = {2019}
}
```
## Acknowledgements
This code is built on [RESCAN](https://github.com/XiaLiPKU/RESCAN) and [TernausNet](https://github.com/thstkdgus35/EDSR-PyTorch). We are particularly grateful to [Xia Li](https://github.com/XiaLiPKU) for the support.

