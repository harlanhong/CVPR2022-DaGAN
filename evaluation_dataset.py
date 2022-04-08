"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from PIL import ImageFile
from skimage import io, img_as_float32
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from data.image_folder import make_dataset
# from PIL import Image
import os
import torch
import pdb
import pandas as pd

class EvaluationDataset(Dataset):
    """A template dataset class for you to implement custom datasets."""
  
    def __init__(self, dataroot, pairs_list=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        # get the image paths of your dataset;
        self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.dataroot = dataroot
        # self.videos = self.videos[5000]
        self.frame_shape = (3,256,256)
        test_videos = os.listdir(os.path.join(self.dataroot,'test'))
        self.videos = test_videos
        pairs = pd.read_csv(pairs_list)
        self.source = pairs['source'].tolist()
        self.driving = pairs['driving'].tolist()
        # self.pose_anchors = pairs['best_frame'].tolist()
        
        self.transforms = T.Compose([T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path_source = self.source[idx]
        path_driving = self.driving[idx]
        # path_anchor = self.pose_anchors[idx]
        anchor = ''
        source = img_as_float32(io.imread(path_source))
        source = np.array(source, dtype='float32')
        source = torch.tensor(source.transpose((2, 0, 1)))
        
        driving = img_as_float32(io.imread(path_driving))
        driving = np.array(driving, dtype='float32')
        driving = torch.tensor(driving.transpose((2, 0, 1)))

        # anchor = img_as_float32(io.imread(path_anchor))
        # anchor = np.array(anchor, dtype='float32')
        # anchor = torch.tensor(anchor.transpose((2, 0, 1)))

        # source = Image.open(path_source).convert('RGB')
        # driving = Image.open(path_driving).convert('RGB')
        # source = T.ToTensor()(source)
        # driving = T.ToTensor()(driving)
        return {'source': source, 'driving': driving, 'path_source': path_source,'path_driving':path_driving, 'anchor': anchor}
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.source)


