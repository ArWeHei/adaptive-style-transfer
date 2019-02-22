from __future__ import division
import math
import scipy.misc
from scipy.ndimage.filters import gaussian_filter

import numpy as np
from ops import *
import random
import copy


def save_batch(input_painting_batch, input_photo_batch, output_painting_batch, output_photo_batch, filepath):
    """
    Concatenates, processes and stores batches as image 'filepath'.
    Args:
        input_painting_batch: numpy array of size [B x H x W x C]
        input_photo_batch: numpy array of size [B x H x W x C]
        output_painting_batch: numpy array of size [B x H x W x C]
        output_photo_batch: numpy array of size [B x H x W x C]
        filepath: full name with path of file that we save

    Returns:

    """
    def batch_to_img(batch):
        return np.reshape(batch,
                          newshape=(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3]))

    inputs = np.concatenate([batch_to_img(input_painting_batch), batch_to_img(input_photo_batch)],
                            axis=0)
    outputs = np.concatenate([batch_to_img(output_painting_batch), batch_to_img(output_photo_batch)],
                             axis=0)

    to_save = np.concatenate([inputs,outputs], axis=1)
    to_save = np.clip(to_save, a_min=0., a_max=255.).astype(np.uint8)

    scipy.misc.imsave(filepath, arr=to_save)


def normalize_arr_of_imgs(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return arr/127.5 - 1.
    # return (arr - np.mean(arr)) / np.std(arr)


def denormalize_arr_of_imgs(arr):
    """
    Inverse of the normalize_arr_of_imgs function.
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return (arr + 1.) * 127.5

def get_conv_layer_shape(input_shape, field_size, padding, stride):
    return np.trunc((input_shape - field_size + 2*padding) / stride + 1)

def get_embedding_size(input_shape):
    field_size = np.array([3, 3])
    initial_padding = np.array([15, 15])

    first_conv = get_conv_layer_shape(input_shape, field_size, initial_padding, 1)
    #print("1st conv: {0}".format(first_conv))
    second_conv = get_conv_layer_shape(first_conv, field_size, 0, 2)
    #print("2nd conv: {0}".format(second_conv))
    third_conv = get_conv_layer_shape(second_conv, field_size, 0, 2)
    #print("3rd conv: {0}".format(third_conv))
    fourth_conv = get_conv_layer_shape(third_conv, field_size, 0, 2)
    #print("4th conv: {0}".format(fourth_conv))
    final_conv = get_conv_layer_shape(fourth_conv, field_size, 0, 2)
    return np.append(final_conv, 256).astype(int)

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots_per_dim = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots_per_dim, img_w * n_plots_per_dim, 3))
    
    for i in range(n_plots_per_dim):
        for j in range(n_plots_per_dim):
            curr_no = i * n_plots_per_dim + j
            if curr_no < images.shape[0]:
                this_img = images[curr_no]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

def center_crop(image, crop_size=(256, 256)):
    rows, cols, chs = image.shape
    #return a croped image around the center
    #since most important information is usually in the center
    if rows < crop_size[1]:
        alpha = float(crop_size[1] / rows)
        #we prefer 1280x768 landscape pictures over portait aspect ratio
        #rows should actually be called row as it's the coordinates in a row
        image = scipy.misc.imresize(arr=image, size=alpha)
        rows, cols, chs = image.shape
    if cols < crop_size[0]:
        alpha = float(crop_size[0] / cols)
        #take the base length as this is usually the longer one
        #we prefer 1280x768 landscape pictures over portait aspect ratio
        #rows should actually be called row as it's the coordinates in a row
        image = scipy.misc.imresize(arr=image, size=alpha)
        rows, cols, chs = image.shape

    #center coordinates
    x = int((rows - crop_size[1])/2)
    y = int((cols - crop_size[0])/2)

    image = image[x:x+crop_size[1], y:y+crop_size[0], :]
    # If the input image was too small to comprise patch of size crop_size,
    # resize obtained patch to desired size.
    return image
    
