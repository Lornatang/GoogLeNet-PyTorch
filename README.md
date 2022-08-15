# GoogleNet-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842v1.pdf).

## Table of contents

- [GoogleNet-PyTorch](#googlenet-pytorch)
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
        - [Going Deeper with Convolutions](#going-deeper-with-convolutions)

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

- line 29: `model_arch_name` change to `googlenet`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `test`.
- line 88: `model_weights_path` change to `./results/pretrained_models/GOOGLENET-ImageNet_1K-64f6524f.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `googlenet`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `train`.
- line 47: `pretrained_model_weights_path` change to `./results/pretrained_models/GoogleNet-ImageNet_1K-32d70693.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `googlenet`.
- line 31: `model_num_classes` change to `1000`.
- line 33: `mode` change to `train`.
- line 50: `resume` change to `./samples/GOOGLENET-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1409.4842v1.pdf](https://arxiv.org/pdf/1409.4842v1.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|   Model   |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:---------:|:-----------:|:-----------------:|:-----------------:|
| GoogleNet | ImageNet_1K |   -(**30.2%**)    | 6.67%(**10.45%**) |

```bash
# Download `GoogleNet-ImageNet_1K-32d70693.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `googlenet` model successfully.
Load `googlenet` model weights `/GoogleNet-PyTorch/results/pretrained_models/GoogleNet-ImageNet_1K-32d70693.pth.tar` successfully.
tench, Tinca tinca                                                          (90.46%)
armadillo                                                                   (2.23%)
barracouta, snoek                                                           (0.70%)
platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus (0.26%)
mud turtle                                                                  (0.17%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Going Deeper with Convolutions

*Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent
Vanhoucke, Andrew Rabinovich*

##### Abstract

We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the
new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (
ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the
network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network
while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the
Hebbian principle and the intuition of multiscale processing. One particular incarnation used in our submission for
ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of
classification and detection.

[[Paper]](https://arxiv.org/pdf/1409.4842v1.pdf)

```bibtex
@inproceedings{szegedy2015going,
  title={Going deeper with convolutions},
  author={Szegedy, Christian and Liu, Wei and Jia, Yangqing and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1--9},
  year={2015}
}
```