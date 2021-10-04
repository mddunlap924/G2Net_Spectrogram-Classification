# Spectrogram Classification



This is code [I (i.e., Myles Dunlap)](https://www.kaggle.com/dunlap0924) developed for the [Kaggle G2Net Gravitational Wave Competition](https://www.kaggle.com/c/g2net-gravitational-wave-detection/overview). Thanks to teammates [Johnny Lee](https://www.kaggle.com/wuliaokaola), [Alex](https://www.kaggle.com/lihuajing), and [Coolz](https://www.kaggle.com/cooolz) we were able to finish in 10th position out of 1,219 teams (i.e., [top 1%](https://www.kaggle.com/dunlap0924)) through creative problem solving and advanced solutions.

The objective of this competition was to determine if a linear sine-sweep signal, also referred to as a chirp waveform, was either present or not in a data instance. The challenge is that the chirp waveform would have a very low signal-to-noise ratio (SNR) and frequency content could change between data instances. SNR is one, if not the most, informative measures of signal detectability. Fundamentally, this competition was a multi-channel signal detection challenge.

The following image shows a chirp waveform that can easily be seen in all three channels of data. In this application there are three different channels of data where each data channel is representative of gravitational wave observatory such as Livingston, Hanford, and Virgo. The signal is present in each of the three but it contains time shifts and different SNR levels.

![](https://github.com/mddunlap924/G2Net_Spectrogram-Classification/blob/main/CQT_Spectrogram.png)

In this repository the time domain signals are converted into spectrograms using a [Constant-Q transform (CQT)](https://en.wikipedia.org/wiki/Constant-Q_transform). A probability from 0-1 for the detection of the chirp waveform is provided for a given data instance. Common and widely used image classification models, obtained from the [timm](https://github.com/rwightman/pytorch-image-models) package, are trained on the CQT images. The [nnAudio](https://github.com/KinWaiCheuk/nnAudio) processing toolbox was used to significantly speed up spectrogram generation for on-the-fly processing. We ensemble this 1D-CNN with other two-dimensional (2D) spectrogram image classification techniques to boost our score. I also provide a 1D-CNN neural network approach for this competition [here](https://github.com/mddunlap924/G2Net_1D-CNN).

# Requirements

The required packages are provided in the [requirements.txt](https://github.com/mddunlap924/G2Net_1D-CNN/blob/main/requirements.txt)

The data is ~77GB and can be found on [Kaggle](https://www.kaggle.com/c/g2net-gravitational-wave-detection/data). 

# Spectrogram Classification

In this section a description of the modeling approach and steps required to execute the code are provided. [PyTorch Lightning][1] was used for this project. 

1. The [execute_models_bash]() is used to execute a model and its configuration. Run commands in the terminal to execute. This file can be modified to execute over multiple configuration files. This approach is helpful for experimenting with various hyperparameters, training on multiple data folds,  and allowing you to work on other tasks. At the time of writing PyTorch Lightning had an issue with [releasing RAM][2] and this was a suitable workaround.

   ```
   sh execute_models_bash
   ```

2. [main.py]() contains the high-level workflow and structure for calling commands such as data loading, k-fold splitting, data preprocessing, training, logging results, and inference. This file provides the overall workflow. 

3. [pl_dataset_model.py]() contains methods and classes for tasks such as data normalization, waveform augmentations, data loaders, data modules, model description, and checkpoint locations.

4. [helper_functions.py]() contains methods and classes for tasks such as logging data with [Weights & Biases][3], signal processing and filtering techniques with [GWpy][4] such as data [spectral whitening][5], loading configuration parameters, and measuring descriptive statistics on the datasets.

The architecture can easily be changed by assigning different models in the [configuration file]().

```
  """ Get Timm Model Config Info """
    model_cfg = timm.create_model(cfg['MODEL']['name']).default_cfg
```

Through multiple experiments is was found that only a few augmentation techniques provide a considerable boost the cross-validation scores and leader board score. Image augmentation was conducted using [torchvision](https://pytorch.org/vision/stable/index.html) and a few of the most beneficial image augmentation techniques were [Resize](https://pytorch.org/vision/0.8/_modules/torchvision/transforms/transforms.html#Resize) and [RandomErasing](https://pytorch.org/vision/0.8/_modules/torchvision/transforms/transforms.html#RandomErasing) .

Users are encouraged to modify the files as they see fit to best work with their applications. 

