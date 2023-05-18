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
import random## import random module that allows generating random values,etc.

import numpy as np## import module, alias as np, that is used for computing and working with arrays
import torch## import torch library
from torch.backends import cudnn## import NVIDIA CUDA deep neural network library

# Random seed to maintain reproducible results
random.seed(0)## initialize with 0 the random number generator
torch.manual_seed(0)## set the seed to 0 for generating random numbers
np.random.seed(0)## set the seed to 0 for generating random numbers
# Use GPU for training by default
device = torch.device("cuda", 0)## create torch.device object representing the first available GPU device
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True## enable setting to automatically find the best algorithm for a specific configuration
# Model arch name
model_arch_name = "googlenet"## initialize variable with string googlenet
# Model number class
model_num_classes = 1000## initialize the number of model classes to be 1000
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name.upper()}-ImageNet_1K"## concatenate uppercase value of model_arch_name and string "-ImageNet_1K"

if mode == "train":
    # Dataset address
    train_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_train"## set train image path
    valid_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"## set valid image path

    image_size = 224## initialize image size to 224
    batch_size = 128## initialize batch size to 128
    num_workers = 4## initialize number of workers(subprocesses) to 4

    # The address to load the pretrained model
    pretrained_model_weights_path = "./results/pretrained_models/GoogleNet-ImageNet_1K-32d70693.pth.tar"## set path of pretrained weights of the model

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 600## initialize number of epochs to 600(number of times the entire dataset will be passed forward and backward through the neural network during the training process)

    # Loss parameters
    loss_label_smoothing = 0.1## initialize to 0.1(instead of using one-hot encoding for the ground truth labels (i.e., 1 for the correct class and 0 for all other classes), we assign a small probability to incorrect classes as well)
    loss_aux3_weights = 1.0## initialize to 1.0
    loss_aux2_weights = 0.3## initialize to 0.3
    loss_aux1_weights = 0.3## initialize to 0.3

    # Optimizer parameter
    model_lr = 0.1## initialize learning rate to 0.1
    model_momentum = 0.9## initialize momentum to 0.9
    model_weight_decay = 2e-05## initialize to 0.00002
    model_ema_decay = 0.99998## initialize exponential moving average decay to 0.99998

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4## set learning rate to one quarter of the number of training epochs(learning rate will drop after the first epochs // 4)
    lr_scheduler_T_mult = 1## set to 1 which means that the learning rate drops will happen at fixed intervals
    lr_scheduler_eta_min = 5e-5## minimum learning rate is set to 0.00005

    # How many iterations to print the training/validate result
    train_print_frequency = 200## print training status every 200 batches during training
    valid_print_frequency = 20## print info about validation process every 200 batches

if mode == "test":
    # Test data address
    test_image_dir = "./data/ImageNet_1K/ILSVRC2012_img_val"## path of test image

    # Test dataloader parameters
    image_size = 224## set image size to 224
    batch_size = 256## set batch size tp 256
    num_workers = 4## set number of workers to 4

    # How many iterations to print the testing result
    test_print_frequency = 20## print test results every 20 iterations

    model_weights_path = "./results/pretrained_models/GoogleNet-ImageNet_1K-32d70693.pth.tar"
