# GoogLeNet-PyTorch

### Update (Feb 17, 2020)

The update is for ease of use and deployment.

 * [Example: Export to ONNX](#example-export-to-onnx)
 * [Example: Extract features](#example-feature-extraction)
 * [Example: Visual](#example-visual)

It is also now incredibly simple to load a pretrained model with a new number of classes for transfer learning:

```python
from googlenet_pytorch import GoogLeNet 
model = GoogLeNet.from_pretrained('googlenet')
```

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained GoogLeNet models 
 * Use VGGNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an GoogLeNet on your own dataset
 * Export GoogLeNet models for production
 
### Table of contents
1. [About GoogLeNet](#about-googlenet)
2. [Installation](#installation)
3. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export-to-onnx)
    * [Example: Visual](#example-visual)
4. [Contributing](#contributing) 

### About GoogLeNet

If you're new to GoogLeNet, here is an explanation straight from the official PyTorch implementation: 

We propose a deep convolutional neural network architecture codenamed "Inception", 
which was responsible for setting the new state of the art for classification and 
detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). 
The main hallmark of this architecture is the improved utilization of the computing 
resources inside the network. This was achieved by a carefully crafted design that allows 
for increasing the depth and width of the network while keeping the computational budget 
constant. To optimize quality, the architectural decisions were based on the Hebbian 
principle and the intuition of multi-scale processing. One particular incarnation used 
in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality 
of which is assessed in the context of classification and detection.

### Installation

Install from pypi:
```bash
$ pip install googlenet_pytorch
```

Install from source:
```bash
$ git clone https://github.com/Lornatang/GoogLeNet-PyTorch.git
$ cd GoogLeNet-PyTorch
$ pip install -e .
``` 

### Usage

#### Loading pretrained models

Load a pretrained GoogLeNet: 
```python
from googlenet_pytorch import GoogLeNet
model = GoogLeNet.from_pretrained("googlenet")
```

Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  googlenet	  |  30.22	    |  10.47      |

#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

Here's a sample execution.

```python
import json

import torch
import torchvision.transforms as transforms
from PIL import Image

from googlenet_pytorch import GoogLeNet 

# Open image
input_image = Image.open("img.jpg")

# Preprocess image
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Load class names
labels_map = json.load(open("labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with GoogLeNet
model = GoogLeNet.from_pretrained("googlenet")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
  input_batch = input_batch.to("cuda")
  model.to("cuda")

with torch.no_grad():
  logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
  label = labels_map[idx]
  prob = torch.softmax(logits, dim=1)[0, idx].item()
  print(f"{label:<75} ({prob * 100:.2f}%)")
```

#### Example: Feature Extraction 

You can easily extract features with `model.extract_features`:
```python
import torch
from googlenet_pytorch import GoogLeNet 
model = GoogLeNet.from_pretrained('googlenet')

# ... image preprocessing as in the classification example ...
inputs = torch.randn(1, 3, 224, 224)
print(inputs.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(inputs)
print(features.shape) # torch.Size([1, 1024, 7, 7])
```

#### Example: Export to ONNX  

Exporting to ONNX for deploying to production is now simple: 
```python
import torch 
from googlenet_pytorch import GoogLeNet 

model = GoogLeNet.from_pretrained('googlenet')
dummy_input = torch.randn(16, 3, 224, 224)

torch.onnx.export(model, dummy_input, "demo.onnx", verbose=True)
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10002/](http://127.0.0.1:10002/).

Enjoy it.

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

For more datasets result. Please see `research/README.md`.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Going Deeper with Convolutions

*Christian Szegedy1, Wei Liu2, Yangqing Jia1, Pierre Sermanet1, Scott Reed3, Dragomir Anguelov1, Dumitru Erhan1, Vincent Vanhoucke1, Andrew Rabinovich4*

##### Abstract

We propose a deep convolutional neural network architecture codenamed Inception that achieves the new
state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014
(ILSVRC14). The main hallmark of this architecture is the
improved utilization of the computing resources inside the
network. By a carefully crafted design, we increased the
depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and
the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC14 is called
GoogLeNet, a 22 layers deep network, the quality of which
is assessed in the context of classification and detection.

[paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)

```text
@article{AlexNet,
title:{Going Deeper with Convolutions},
author:{Christian Szegedy1, Wei Liu2, Yangqing Jia1, Pierre Sermanet1, Scott Reed3, Dragomir Anguelov1, Dumitru Erhan1, Vincent Vanhoucke1, Andrew Rabinovich4},
journal={cvpr},
year={2015}
}
```