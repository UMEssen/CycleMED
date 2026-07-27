"""This module contains simple helper functions """
from __future__ import print_function
import numpy as np
from PIL import Image
import os
import cv2
import torch

def tensor2im(input_image, imtype=np.uint16, scale_to_hu=False, verbose=False):
    """Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor or ndarray) -- the input image tensor or numpy array
        imtype (type) -- the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # Check if the input is a tensor
            image_tensor = input_image.data
        else:
            return input_image

        image_numpy = image_tensor[0].cpu().float().numpy()  # Convert the tensor to a numpy array
        if verbose:
            print(f"Converted tensor with shape {image_tensor.shape}, dtype {image_tensor.dtype}")
            print(f"Image range before scaling: min={image_numpy.min()}, max={image_numpy.max()}")

        if scale_to_hu:
            # Scaling from [-1, 1] to Hounsfield Units [-1000, 3000]
            image_numpy = image_numpy * 2000 + 1000
            image_numpy = (image_numpy + 1000) / 4000 * 65535
        else:
            # Normalizing to [0, 65535] directly from [-1, 1]
            image_numpy = (image_numpy + 1) / 2 * 65535

        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # Transpose to match height x width x channels
        if verbose:
            print(f"Image range after scaling: min={image_numpy.min()}, max={image_numpy.max()}")
    else:
        image_numpy = input_image

    image_numpy = image_numpy.astype(imtype)
    return image_numpy


# def tensor2im(input_image, imtype=np.uint8):
#     """"Converts a Tensor array into a numpy image array.

#     Parameters:
#         input_image (tensor) --  the input image tensor array
#         imtype (type)        --  the desired type of the converted numpy array
#     """
#     if not isinstance(input_image, np.ndarray):
#         if isinstance(input_image, torch.Tensor):  # get the data from a variable
#             image_tensor = input_image.data
#         else:
#             return input_image
#         image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
#         if image_numpy.shape[0] == 1:  # grayscale to RGB
#             image_numpy = np.tile(image_numpy, (3, 1, 1))
#         image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
#     else:  # if it is a numpy array, do nothing
#         image_numpy = input_image
#     return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    h, w = image_numpy.shape[:2]

    # Resizing the image according to the aspect ratio
    if aspect_ratio > 1.0:
        resized_width = int(w * aspect_ratio)
        image_numpy = cv2.resize(image_numpy, (resized_width, h), interpolation=cv2.INTER_CUBIC)
    elif aspect_ratio < 1.0:
        resized_height = int(h / aspect_ratio)
        image_numpy = cv2.resize(image_numpy, (w, resized_height), interpolation=cv2.INTER_CUBIC)

    # Save the image using cv2
    cv2.imwrite(image_path, image_numpy)
# def save_image(image_numpy, image_path, aspect_ratio=1.0):
#     """Save a numpy image to the disk

#     Parameters:
#         image_numpy (numpy array) -- input numpy array
#         image_path (str)          -- the path of the image
#     """

#     image_pil = Image.fromarray(image_numpy)
#     h, w, _ = image_numpy.shape

#     print(f"Original image shape: {image_numpy.shape}, Type: {image_numpy.dtype}")

#     if aspect_ratio > 1.0:
#         new_height = h
#         new_width = int(w * aspect_ratio)
#         image_pil = image_pil.resize((new_width, new_height), Image.BICUBIC)
#         print(f"Resizing image to new dimensions: {new_width}x{new_height} (width increased)")
#     elif aspect_ratio < 1.0:
#         new_height = int(h / aspect_ratio)
#         new_width = w
#         image_pil = image_pil.resize((new_width, new_height), Image.BICUBIC)
#         print(f"Resizing image to new dimensions: {new_width}x{new_height} (height increased)")

#     # if aspect_ratio > 1.0:
#     #     image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
#     # if aspect_ratio < 1.0:
#     #     image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
#     # image_pil.save(image_path)
#     image_pil.save(image_path)
#     print(f"Image saved to: {image_path}")

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def compute_visual_difference(image1, image2):
    # Compute absolute difference and scale to full range of uint16 for visibility
    difference = np.abs(image1.astype(np.int32) - image2.astype(np.int32))
    normalized_difference = np.clip(difference, 0, 65535).astype(np.uint16)
    return normalized_difference
