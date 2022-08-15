# VGG-PyTorch

<a href="https://console.tiyaro.ai/explore/trn:model:123456789012-venkat:1.0:alexnet_pytorch_6c50c5">
<img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/tiyaro_badge.svg"></a>

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556v6.pdf).

## Table of contents

- [VGG-PyTorch](#vgg-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Very Deep Convolutional Networks for Large-Scale Image Recognition](#very-deep-convolutional-networks-for-large-scale-image-recognition)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `vgg11`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `test`.
- line 81: `model_weights_path` change to `./results/pretrained_models/VGG11-ImageNet_1K-64f6524f.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `vgg11`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `train`.
- line 47: `pretrained_model_weights_path` change to `./results/pretrained_models/VGG11-ImageNet_1K-64f6524f.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `vgg11`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `train`.
- line 50: `resume` change to `./samples/VGG11-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1409.1556v6.pdf](https://arxiv.org/pdf/1409.1556v6.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|  Model   |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:--------:|:-----------:|:-----------------:|:-----------------:|
|  VGG11   | ImageNet_1K | 29.6%(**30.9%**)  | 10.4%(**11.3%**)  |
| VGG11_BN | ImageNet_1K |   -(**29.6%**)    |   -(**10.2%**)    |
|  VGG13   | ImageNet_1K | 28.7%(**30.1%**)  |  9.9%(**10.8%**)  |
| VGG13_BN | ImageNet_1K |   -(**28.4%**)    |    -(**9.6%**)    |
|  VGG16   | ImageNet_1K | 27.0%(**28.4%**)  |  8.8%(**9.6%**)   |
| VGG16_BN | ImageNet_1K |   -(**26.6%**)    |    -(**8.5%**)    |
|  VGG19   | ImageNet_1K | 27.3%(**27.6%**)  |  9.0%(**9.1%**)   |
| VGG19_BN | ImageNet_1K |   -(**25.7%**)    |    -(**8.1%**)    |

```bash
# Download `VGG11-ImageNet_1K-64f6524f.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build VGG11 model successfully.
Load VGG11 model weights `/VGG-PyTorch/results/pretrained_models/VGG11-ImageNet_1K-64f6524f.pth.tar` successfully.
tench, Tinca tinca                                                          (74.97%)
barracouta, snoek                                                           (23.09%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.81%)
reel                                                                        (0.45%)
armadillo                                                                   (0.25%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Very Deep Convolutional Networks for Large-Scale Image Recognition

*Karen Simonyan, Andrew Zisserman*

##### Abstract

In this work we investigate the effect of the convolutional network depth on its
accuracy in the large-scale image recognition setting. Our main contribution is
a thorough evaluation of networks of increasing depth using an architecture with
very small (3×3) convolution filters, which shows that a significant improvement
on the prior-art configurations can be achieved by pushing the depth to 16–19
weight layers. These findings were the basis of our ImageNet Challenge 2014
submission, where our team secured the first and the second places in the localisation and classification tracks
respectively. We also show that our representations
generalise well to other datasets, where they achieve state-of-the-art results. We
have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep
visual representations in computer vision.

[[Paper]](https://arxiv.org/pdf/1409.1556v6.pdf)

```bibtex
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```