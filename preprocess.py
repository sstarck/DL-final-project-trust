#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
import random
import os
import cv2

def get_data(directory, path, attribute):
    """
    Inputs: directory (the path to the image data), path (the path to image labels), attribute (the factor looking for, e.g. trustworthiness)
    Outputs: x (the image data), y (labels)
    """
    labels = pd.read_csv(path, delim_whitespace=False, header=0, index_col=0)
    data = os.listdir(directory) # return a list of image files
    x = []
    y = []
    batch_size = 100 # training data contains ~5000 images --> memory error if don't process in batches
    iters = int ( len(data) / batch_size )
    for i in range(iters):
        batchData = data[(i * batch_size):(i * batch_size + batch_size)]
        bx = []
        for image_path in batchData:
            image = cv2.imread(os.path.join(directory, image_path)) # Load image from file path
            image = image.astype('float32') / 255 # Normalize
            bx.append(image)
            y.append(labels[attribute][image_path]) # Get the label associated with that image
        np.array(bx)
        x.append(bx)
    z = np.concatenate(x) # Gather numpy array of each batch into one large dataset
    y = np.array(y)
    z = z.astype('float64') # cast to float64 so works with convolution layers
    y = y.astype('float64')
    return z, y

def random_noise(image, space):
    """
    Inputs: image, space (the paper's hyperparameters)
    Output: the image with random noise added
    """
    noise = random.uniform(space["noise_max_freq_pct"], space["noise_min_freq_pct"]) / 100 # Use published hyperparameters to determine noise level
    mask = np.random.uniform(0, 1, (150, 130)) # Create random mask of size (height, width)
    image[np.where(mask < noise)] = 0 # Add noise (i.e. drops color) whenever probability distribution in mask falls within given noise hyperparameter
    return image

def brightness(image, space):
    """
    Inputs: image, space (the paper's hyperparameters)
    Output: the image with brightness increased
    """
    bright_min, bright_max = int(space['brighten_extent_min']), int(space['brighten_extent_max']) # Get hyperparameters and cast to integer
    brightness = random.randint(bright_min, bright_max) / 100 # Choose brightness parameter randomly from within hyperparameter range
    height, width = image.shape[:2] # Should be (150,130)
    for x in range(height): # Add brightness to image
        for y in range(width):
            val = image[x, y, 0][0] + brightness # Potential new color value
            if (0 <= val <= 1): # Ensure stay within [0,1] bounds (data has already been normalized from (0,255) scale)
                image[x, y, :] += val
            elif val < 0:
                image[x, y, :] = [0,0,0] # Since black & white, all 3 RBG values are the same in these images
            elif val > 1:
                image[x, y, :] = [1,1,1]
    return image

def flip(image):
    """
    Inputs: image
    Output: the image flipped left to right (horizontally)
    """
    return np.fliplr(image)

