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
import queue## import queue module
import sys## import sys module which lets us access system-specific parameters and functions
import threading## import threading module which allows us to create and manage new threads of execution
from glob import glob## import function glob from module glob which returns all file paths that match a specific pattern

import cv2## import opencv module which allows us to perform image processing and computer vision tasks
import torch## import torch library which allows us to work with deep neural networks
from PIL import Image## import Image module from PIL module 
from torch.utils.data import Dataset, DataLoader## import modules that load and handle data during the training of deep learning models
from torchvision import transforms## import module that provides a set of common image transformations that can be used to preprocess or augment image data
from torchvision.datasets.folder import find_classes## import function that is used to find folder names in a given directory
from torchvision.transforms import TrivialAugmentWide## import dataset-independent data-augmentation

import imgproc## import imgproc module

__all__ = [
    "ImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")## tuple of supported file extensions used for image files

# The delimiter is not the same between different platforms
if sys.platform == "win32":## check if operating system is Windows
    delimiter = "\\"## use "\\" when delimitating
else:
    delimiter = "/"## use "/" when delimitating


class ImageDataset(Dataset):## define class ImageDataset that inherits from Dataset
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:## define constructor that takes as arguments the dataset address, image size and mode
        super(ImageDataset, self).__init__()## call constructor of superclass
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")## find all image paths that are in any subdirectory of image_dir
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)## discard the first element of the tuple(the list of image paths) and store the 2nd element(dictionary mapping class names to class indices)  
        self.image_size = image_size## assign the image size
        self.mode = mode## assign the mode
        self.delimiter = delimiter## assign the delimiter

        if self.mode == "Train":## if we are in training mode
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## compose several transforms together
                transforms.RandomResizedCrop(self.image_size),## crop a random portion of image and resize it to the given size
                TrivialAugmentWide(),## data-augmentation
                transforms.RandomRotation([0, 270]),## rotate image with a degree randomly selected from the range
                transforms.RandomHorizontalFlip(0.5),## horizontally flip the given image randomly with the given probability
                transforms.RandomVerticalFlip(0.5),## vertically flip the given image randomly with the given probability
            ])
        elif self.mode == "Valid" or self.mode == "Test":## if we are in a valid or test mode
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([## compose several transforms together
                transforms.Resize(256),## resize the input image to the given size
                transforms.CenterCrop([self.image_size, self.image_size]),## crops the crops the given image at the center(square crop with size image_size)
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"## else raise an exception

        self.post_transform = transforms.Compose([## compose several transform together
            transforms.ConvertImageDtype(torch.float),## convert a tensor image to the type float and scale the values accordingly
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])## normalize a tensor image with mean and standard deviation for 3 channels
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:## method that takes an integer batch_index and returns a tuple of a tensor and an index(label index for the class the image belongs to)
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]## get path of image using batch_index and extract from it the directory path and image name 
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:## get file extension, then lowercase it and check if it can be found in the list IMG_EXTENSIONS(file name string is split at period(.))
            image = cv2.imread(self.image_file_paths[batch_index])## read image where the path is accessed in the list of paths using batch index
            target = self.class_to_idx[image_dir]## retrieve the index of the class to which the current image belongs based on the directory of the image
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## convert image from BGR(blue-green-red) to RGB(red-green-blue)

        # OpenCV convert PIL
        image = Image.fromarray(image)## convert above image to a 'PIL' image

        # Data preprocess
        image = self.pre_transform(image)## apply pre_transform function on the image

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False)## convert image to tensor

        # Data postprocess
        tensor = self.post_transform(tensor)## apply post_transform functions on the image

        return {"image": tensor, "target": target}## return list of 2 tuples, each containing a tensor(representing the image) and an int(representing the target class index)

    def __len__(self) -> int:
        return len(self.image_file_paths)## return the number of image paths


class PrefetchGenerator(threading.Thread):## define class PrefetchGenerator that inherits from the thread module
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:## constructor that takes 2 parameters: generator(data generator object that will produce batches of data) and an integer that represents the number of data batches that should be loaded in advance
        threading.Thread.__init__(self)## call constructor of superclass
        self.queue = queue.Queue(num_data_prefetch_queue)## undefined
        self.generator = generator## initialize generator
        self.daemon = True
        self.start()## undefined

    def run(self) -> None:## undefined
        for item in self.generator:## undefined
            self.queue.put(item)## undefined
        self.queue.put(None)## undefined

    def __next__(self):
        next_item = self.queue.get()## undefined
        if next_item is None:## undefined
            raise StopIteration## undefined
        return next_item## undefined

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):## undefined
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue## undefined
        super(PrefetchDataLoader, self).__init__(**kwargs)## undefined

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)## undefined


class CPUPrefetcher:## undefined
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:## undefined
        self.original_dataloader = dataloader## undefined
        self.data = iter(dataloader)## undefined

    def next(self):
        try:## undefined
            return next(self.data)## undefined
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)## undefined

    def __len__(self) -> int:
        return len(self.original_dataloader)## undefined


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):## undefined
        self.batch_data = None## undefined
        self.original_dataloader = dataloader## undefined
        self.device = device## undefined

        self.data = iter(dataloader)## undefined
        self.stream = torch.cuda.Stream()## undefined
        self.preload()

    def preload(self):## undefined
        try:## undefined
            self.batch_data = next(self.data)## undefined
        except StopIteration:## undefined
            self.batch_data = None## undefined
            return None## undefined

        with torch.cuda.stream(self.stream):## undefined
            for k, v in self.batch_data.items():## undefined
                if torch.is_tensor(v):## undefined## undefined
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):## undefined
        torch.cuda.current_stream().wait_stream(self.stream)## undefined
        batch_data = self.batch_data## undefined
        self.preload()## undefined
        return batch_data## undefined

    def reset(self):## undefined
        self.data = iter(self.original_dataloader)## undefined
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
