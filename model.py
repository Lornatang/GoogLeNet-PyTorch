# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from collections import namedtuple## import named tuple type; fields can be accessed by name instead of position index
from typing import Optional, Tuple, Any## import optional(Optional[X]=X | None), tuple(of two items) and any(possible to perform any operation or method call on a value of type Any and assign it to any variable) type

import torch## import torch library
from torch import Tensor## import vector data type from torch library
from torch import nn## import neural network things such as various neural network layer types

__all__ = [## define and initialize a list of strings called __all__
    "GoogLeNetOutputs",
    "GoogLeNet",
    "BasicConv2d", "Inception", "InceptionAux",
    "googlenet",
]

# According to the writing of the official library of Torchvision
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])## create a tuple called "GoogLeNetOutputs" with the same typename and field names:logits, aux_logits2, aux_logits1
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}## dictionary variable that keeps pairs of (key,value)


class GoogLeNet(nn.Module):## define class GoogLeNet that subclasses the base class for all neural network modules
    __constants__ = ["aux_logits", "transform_input"]## define a list

    def __init__(
            self,
            num_classes: int = 1000,## define an int called num_classes that takes the value 1000
            aux_logits: bool = True,## define a bool called aux_logits that takes the value True
            transform_input: bool = False,## define a bool called transform_input that takes the value False
            dropout: float = 0.2,## define a float called dropout that takes the value 0.2
            dropout_aux: float = 0.7,## define a float called dropout_aux that takes the value 0.7
    ) -> None:
        super(GoogLeNet, self).__init__()## call the __init__ method of the superclass
        self.aux_logits = aux_logits## initialize field "aux_logits" of current class
        self.transform_input = transform_input## initialize field "transform_input" of current class

        self.conv1 = BasicConv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))## initialize with 2D convolutional layer having:
        """3 input channels
            64 ouput channels
            7x7 kernel used by the layer
            stride=(2,2) means the kernel will move by 2 pixels horizontally and vertically across the input in each step
            padding=(3,3) means a padding of 3 pixels will be added on each side of the input"""
        self.maxpool1 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)## apply a 2D max pooling over input
        """= downsampling operation in its input tensor by dividing it into small rectangular regions 
            and taking the maximum value within each region
           (3,3) - the size of the pooling window(kernel)
           (2,2) - the stride of the pooling operation(step size)
           ceil_mode=True - use ceil(instead of floor) when computing output shape"""
        self.conv2 = BasicConv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))## initialize with 2D convolutional layer, see more details at line 49
        self.conv3 = BasicConv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))## initialize with 2D convolutional layer, see more details at line 49
        self.maxpool2 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)## apply 2D max pooling over input

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)## create instance of Inception class 
        """in_channels = 192,
            ch1x1 = 64,
            ch3x3red = 96,
            ch3x3 = 128,
            ch5x5red = 16,
            ch5x5 = 32,
            pool_proj = 32"""
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)## create instance of Inception class 
        self.maxpool3 = nn.MaxPool2d((3, 3), (2, 2), ceil_mode=True)## apply 2D max pooling over input

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)## create instance of Inception class 
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)## create instance of Inception class 
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)## create instance of Inception class 
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)## create instance of Inception class 
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)## create instance of Inception class 
        self.maxpool4 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)## apply 2D max pooling over input

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)## create instance of Inception class 
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)## create instance of Inception class 

        if aux_logits:## undefined
            self.aux1 = InceptionAux(512, num_classes, dropout_aux)## undefined
            self.aux2 = InceptionAux(528, num_classes, dropout_aux)## undefined
        else:
            self.aux1 = None## undefined
            self.aux2 = None## undefined

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))## undefined
        self.dropout = nn.Dropout(dropout, True)## undefined
        self.fc = nn.Linear(1024, num_classes)## undefined

        # Initialize neural network weights
        self._initialize_weights()## undefined

    @torch.jit.unused## undefined
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        if self.training and self.aux_logits:## undefined
            return GoogLeNetOutputs(x, aux2, aux1)## undefined
        else:
            return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:## undefined
        out = self._forward_impl(x)## undefined

        return out

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5## undefined
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5## undefined
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5## undefined
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)## undefined

        out = self.conv1(x)## undefined
        out = self.maxpool1(out)## undefined
        out = self.conv2(out)## undefined
        out = self.conv3(out)## undefined
        out = self.maxpool2(out)## undefined

        out = self.inception3a(out)## undefined
        out = self.inception3b(out)## undefined
        out = self.maxpool3(out)## undefined
        out = self.inception4a(out)## undefined
        aux1: Optional[Tensor] = None## undefined
        if self.aux1 is not None:## undefined
            if self.training:## undefined
                aux1 = self.aux1(out)## undefined

        out = self.inception4b(out)## undefined
        out = self.inception4c(out)## undefined
        out = self.inception4d(out)## undefined
        aux2: Optional[Tensor] = None## undefined
        if self.aux2 is not None:## undefined
            if self.training:## undefined
                aux2 = self.aux2(out)## undefined

        out = self.inception4e(out)## undefined
        out = self.maxpool4(out)## undefined
        out = self.inception5a(out)## undefined
        out = self.inception5b(out)## undefined

        out = self.avgpool(out)## undefined
        out = torch.flatten(out, 1)## undefined
        out = self.dropout(out)## undefined
        aux3 = self.fc(out)## undefined

        if torch.jit.is_scripting():## undefined
            return GoogLeNetOutputs(aux3, aux2, aux1)## undefined
        else:
            return self.eager_outputs(aux3, aux2, aux1)## undefined

    def _initialize_weights(self) -> None:## undefined
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):## undefined
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-2, b=2)## undefined
            elif isinstance(module, nn.BatchNorm2d):## undefined
                nn.init.constant_(module.weight, 1)## undefined
                nn.init.constant_(module.bias, 0)## undefined


class BasicConv2d(nn.Module):## undefined
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()## undefined
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)## undefined

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)## undefined
        out = self.bn(out)## undefined
        out = self.relu(out)## undefined

        return out


class Inception(nn.Module):## define class Inception that subclasses the base class for all neural network modules
    def __init__(
            self,
            in_channels: int,## number of input channels
            ch1x1: int,## number of 1x1 filters
            ch3x3red: int,## number of 1x1 filters in the reduction layer used before the 3x3 convolution
            ch3x3: int,## number of 3x3 filters
            ch5x5red: int,## number of 1x1 filters in the reduction layer used before the 5x5 convolution
            ch5x5: int,## number of 5x5 filters
            pool_proj: int,## number of 1×1 filters in the projection layer after the built-in max-pooling  
    ) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))## undefined

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),## undefined
            BasicConv2d(ch3x3red, ch3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),## undefined
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),## undefined
            BasicConv2d(ch5x5red, ch5x5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),## undefined
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), ceil_mode=True),## undefined
            BasicConv2d(in_channels, pool_proj, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),## undefined
        )

    def forward(self, x: Tensor) -> Tensor:## undefined
        branch1 = self.branch1(x)## undefined
        branch2 = self.branch2(x)## undefined
        branch3 = self.branch3(x)## undefined
        branch4 = self.branch4(x)## undefined
        out = [branch1, branch2, branch3, branch4]## undefined

        out = torch.cat(out, 1)## undefined

        return out


class InceptionAux(nn.Module):## undefined
    def __init__(## undefined
            self,## undefined
            in_channels: int,## undefined
            num_classes: int,## undefined
            dropout: float = 0.7,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))## undefined
        self.conv = BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))## undefined
        self.relu = nn.ReLU(True)## undefined
        self.fc1 = nn.Linear(2048, 1024)## undefined
        self.fc2 = nn.Linear(1024, num_classes)## undefined
        self.dropout = nn.Dropout(dropout, True)## undefined

    def forward(self, x: Tensor) -> Tensor:
        out = self.avgpool(x)## undefined
        out = self.conv(out)## undefined
        out = torch.flatten(out, 1)## undefined
        out = self.fc1(out)## undefined
        out = self.relu(out)## undefined
        out = self.dropout(out)## undefined
        out = self.fc2(out)## undefined

        return out


def googlenet(**kwargs: Any) -> GoogLeNet:## undefined
    model = GoogLeNet(**kwargs)

    return model

