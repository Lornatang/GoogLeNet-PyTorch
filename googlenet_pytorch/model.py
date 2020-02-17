# Copyright 2020 Lorna Authors. All Rights Reserved.
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

""" From pytorch official website """
import warnings
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit.annotations import Optional, Tuple

from .utils import get_model_params
from .utils import googlenet_params
from .utils import load_pretrained_weights

GoogLeNetOutputs = namedtuple(
  'GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNet(nn.Module):
  __constants__ = ['aux_logits', 'transform_input']

  def __init__(self, global_params=None):
    super(GoogLeNet, self).__init__()
    self._global_params = global_params
    aux_logits = self._global_params.aux_logits
    transform_input = self._global_params.transform_input
    blocks = self._global_params.blocks
    dropout_rate = self._global_params.dropout_rate
    num_classes = self._global_params.num_classes

    if blocks is None:
      blocks = [BasicConv2d, Inception, InceptionAux]
    assert len(blocks) == 3
    conv_block = blocks[0]
    inception_block = blocks[1]
    inception_aux_block = blocks[2]

    self.aux_logits = aux_logits
    self.transform_input = transform_input

    self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
    self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    self.conv2 = conv_block(64, 64, kernel_size=1)
    self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
    self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
    self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
    self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
    self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
    self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
    self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
    self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
    self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
    self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

    if aux_logits:
      self.aux1 = inception_aux_block(512, num_classes)
      self.aux2 = inception_aux_block(528, num_classes)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(p=dropout_rate)
    self.fc = nn.Linear(1024, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def _transform_input(self, x):
    # type: (Tensor) -> Tensor
    if self.transform_input:
      x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
      x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
      x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
      x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    return x

  def _forward(self, x):
    # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
    # N x 3 x 224 x 224
    x = self.conv1(x)
    # N x 64 x 112 x 112
    x = self.maxpool1(x)
    # N x 64 x 56 x 56
    x = self.conv2(x)
    # N x 64 x 56 x 56
    x = self.conv3(x)
    # N x 192 x 56 x 56
    x = self.maxpool2(x)

    # N x 192 x 28 x 28
    x = self.inception3a(x)
    # N x 256 x 28 x 28
    x = self.inception3b(x)
    # N x 480 x 28 x 28
    x = self.maxpool3(x)
    # N x 480 x 14 x 14
    x = self.inception4a(x)
    # N x 512 x 14 x 14
    aux_defined = self.training and self.aux_logits
    if aux_defined:
      aux1 = self.aux1(x)
    else:
      aux1 = None

    x = self.inception4b(x)
    # N x 512 x 14 x 14
    x = self.inception4c(x)
    # N x 512 x 14 x 14
    x = self.inception4d(x)
    # N x 528 x 14 x 14
    if aux_defined:
      aux2 = self.aux2(x)
    else:
      aux2 = None

    x = self.inception4e(x)
    # N x 832 x 14 x 14
    x = self.maxpool4(x)
    # N x 832 x 7 x 7
    x = self.inception5a(x)
    # N x 832 x 7 x 7
    x = self.inception5b(x)
    # N x 1024 x 7 x 7

    x = self.avgpool(x)
    # N x 1024 x 1 x 1
    x = torch.flatten(x, 1)
    # N x 1024
    x = self.dropout(x)
    x = self.fc(x)
    # N x 1000 (num_classes)
    return x, aux2, aux1

  @torch.jit.unused
  def eager_outputs(self, x, aux2, aux1):
    # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
    if self.training and self.aux_logits:
      return _GoogLeNetOutputs(x, aux2, aux1)
    else:
      return x

  def extract_features(self, inputs):
    """ Returns output of the final convolution layer """
    x = self.conv1(inputs)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.maxpool2(x)

    x = self.inception3a(x)
    x = self.inception3b(x)
    x = self.maxpool3(x)
    x = self.inception4a(x)

    x = self.inception4b(x)
    x = self.inception4c(x)
    x = self.inception4d(x)

    x = self.inception4e(x)
    x = self.maxpool4(x)
    x = self.inception5a(x)
    x = self.inception5b(x)
    return x

  def forward(self, x):
    # type: (Tensor) -> GoogLeNetOutputs
    x = self._transform_input(x)
    x, aux1, aux2 = self._forward(x)
    aux_defined = self.training and self.aux_logits
    if torch.jit.is_scripting():
      if not aux_defined:
        warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
      return GoogLeNetOutputs(x, aux2, aux1)
    else:
      return self.eager_outputs(x, aux2, aux1)

  @classmethod
  def from_name(cls, model_name, override_params=None):
    cls._check_model_name_is_valid(model_name)
    global_params = get_model_params(model_name, override_params)
    return cls(global_params)

  @classmethod
  def from_pretrained(cls, model_name, num_classes=1000):
    model = cls.from_name(model_name, override_params={"num_classes": num_classes})
    load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
    return model

  @classmethod
  def get_image_size(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    _, _, _, res = googlenet_params(model_name)
    return res

  @classmethod
  def _check_model_name_is_valid(cls, model_name):
    """ Validates model name. None that pretrained weights are only available for
    the first four models (googlenet) at the moment. """
    valid_model = ['googlenet']
    if model_name not in valid_model:
      raise ValueError('model_name should be one of: ' + ', '.join(valid_model))


class Inception(nn.Module):
  __constants__ = ['branch2', 'branch3', 'branch4']

  def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
               conv_block=None):
    super(Inception, self).__init__()
    if conv_block is None:
      conv_block = BasicConv2d
    self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

    self.branch2 = nn.Sequential(
      conv_block(in_channels, ch3x3red, kernel_size=1),
      conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
    )

    self.branch3 = nn.Sequential(
      conv_block(in_channels, ch5x5red, kernel_size=1),
      conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
    )

    self.branch4 = nn.Sequential(
      nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
      conv_block(in_channels, pool_proj, kernel_size=1)
    )

  def _forward(self, x):
    branch1 = self.branch1(x)
    branch2 = self.branch2(x)
    branch3 = self.branch3(x)
    branch4 = self.branch4(x)

    outputs = [branch1, branch2, branch3, branch4]
    return outputs

  def forward(self, x):
    outputs = self._forward(x)
    return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

  def __init__(self, in_channels, num_classes, conv_block=None):
    super(InceptionAux, self).__init__()
    if conv_block is None:
      conv_block = BasicConv2d
    self.conv = conv_block(in_channels, 128, kernel_size=1)

    self.fc1 = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(1024, num_classes)

  def forward(self, x):
    # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
    x = F.adaptive_avg_pool2d(x, (4, 4))
    # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
    x = self.conv(x)
    # N x 128 x 4 x 4
    x = torch.flatten(x, 1)
    # N x 2048
    x = F.relu(self.fc1(x), inplace=True)
    # N x 1024
    x = F.dropout(x, 0.7, training=self.training)
    # N x 1024
    x = self.fc2(x)
    # N x 1000 (num_classes)

    return x


class BasicConv2d(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(BasicConv2d, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return F.relu(x, inplace=True)

#
# def load_pretrained_weights(model_name, global_params):
#   """ Loads pretrained weights, and downloads if loading for the first time. """
#   try:
#     _create_unverified_https_context = ssl._create_unverified_context
#   except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
#   else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = _create_unverified_https_context
#
#   state_dict = model_zoo.load_url(url_map[model_name])
#   model = GoogLeNet(global_params)
#   model.load_state_dict(state_dict, strict=False)
#
#   print(f"Loaded pretrained weights for {model_name}")
#   return model
