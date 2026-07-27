import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import cv2
import numpy as np
from torch import from_numpy

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        print(f"Directory A: {self.dir_A}, Directory B: {self.dir_B}")
        print(f"Total images in A: {self.A_size}, Total images in B: {self.B_size}")
        print(f"Color channels in input: {input_nc}, Color channels in output: {output_nc}")

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        # Load images using OpenCV in the correct format
        A_img = cv2.imread(A_path, cv2.IMREAD_UNCHANGED)
        B_img = cv2.imread(B_path, cv2.IMREAD_UNCHANGED)

        if A_img is None:
            raise RuntimeError(f"Failed to load image A from path: {A_path}")
        if B_img is None:
            raise RuntimeError(f"Failed to load image B from path: {B_path}")
        if self.opt.verbose:
            print(f"Image A Path: {A_path}, Shape: {A_img.shape}, Dtype: {A_img.dtype}")
            print(f"Image B Path: {B_path}, Shape: {B_img.shape}, Dtype: {B_img.dtype}")

        # Convert images to float type and normalize to the range [-1, 1]
        A_img = (A_img.astype(np.float32) / 65535.0) * 2 - 1
        B_img = (B_img.astype(np.float32) / 65535.0) * 2 - 1

        # Add a channel dimension if the images are grayscale
        A = from_numpy(A_img).unsqueeze(0) if A_img.ndim == 2 else from_numpy(A_img).permute(2, 0, 1)
        B = from_numpy(B_img).unsqueeze(0) if B_img.ndim == 2 else from_numpy(B_img).permute(2, 0, 1)

        if self.opt.verbose:
            print(f"Tensor A Shape: {A.shape}, Type: {A.dtype}")
            print(f"Tensor B Shape: {B.shape}, Type: {B.dtype}")

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
