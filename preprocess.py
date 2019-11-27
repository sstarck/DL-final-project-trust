import numpy as np
import tensorflow as tf
import pandas as pd
import random
import os
import cv2 # Need to import opencv
#from PIL import Image # Need to import Pillow

def get_data(directory, annotation, attribute, norm=True):
    """
    Inputs: directory (the path to the image data), annotation (the path to image annotations), attribute (the factor looking for, e.g. trustworthiness)
    Outputs: x (the image data), y (labels)

    NOTE: never use norm
    """
    labels = pd.read_csv(annotation, delim_whitespace=False, header=0, index_col=0)
    # data = filter(lambda x: x in labels.index.values, os.listdir(directory)) - what does this do????
    data = os.listdir(directory) # Should return a list of image files
    # print("shape of data = ", len(data)) # (4874)
    x = []
    y = []
    for image_path in data:
        image = cv2.imread(os.path.join(directory, image_path)) # Load image from file path
        # NORMALIZE IMAGE HERE BY / 255 IF WANT
        # JOHN COMMENTED THIS OUT because dims didn't work with conv2d layer. And because TA said we don't need this
        #image = image[:, :, 0] # Squish RBG to one pixel value since greyscale    
        x.append(image)
        y.append(labels[attribute][image_path]) # Get the label associated with that image
    # Do some kind of normalization on labels???
    #y = np.array(y)
    #y = y - np.min(y)
    #y = np.float32(y / np.max(y))
    x, y = np.array(x), np.array(y)
    
    # JOHN ADDED THIS. x was previously uint8 which didnt work with conv layers 
    x = x.astype('float32')
    y = y.astype('float32')
   
    # print("Shape of x = ", x.shape, " and of y = ", y.shape) # x = (4874, 150, 130) and y = (4874,) <--- FIGURE OUT HOW TO SQUISH TO 1 IN LAST X DIM???
    return x, y

def random_noise(image, space):
    """
    Inputs: image, space (the paper's hyperparameters)
    Output: the image with random noise added
    """
    height, width = image.shape[:2] # Get dimensions of the image
    noise = random.uniform(space["noise_max_freq_pct"], space["noise_min_freq_pct"]) / 100 # Use hyperparameters to determine noise level
    mask = np.random.uniform(0, 1, (height, width)) # Create random mask
    image[np.where(mask < noise)] = 0 # Add noise according to mask
    return image

def brightness(image, space):
    """
    Inputs: image, space (the paper's hyperparameters)
    Output: the image with brightness increased
    """
    bright_min, bright_max = int(space['brighten_extent_min']), int(space['brighten_extent_max']) # Get hyperparameters and cast to integer
    brightness = random.randint(bright_min, bright_max) # Choose brightness parameter randomly from within hyperparameter range
    newImage = image # Check that types are ok - float vs. int
    if brightness > 0: # Add positive brightness
        newImage[np.where(newImage + brightness < 255)] += np.uint8(brightness) # Normal
        newImage[np.where(newImage + brightness >= 255)] = 255 # If would cause to go beyond bounds of RBG
    else: # Add negative brightness
        newImage[np.where(newImage + brightness > 0)] += np.uint8(brightness)
        newImage[np.where(newImage + brightness <= 0)] = 0
    return newImage

def flip(image):
    """
    Inputs: image
    Output: the image flipped left to right (horizontally)
    """
    return np.fliplr(image)

"""
OLD CODE

# Normalize during get data
        if norm:
            image = image.astype('float32') / 255 # Normalize
            # image = cv2.resize(image, (1, 150, 130))
            # image.shape = (1, 150, 130) # Change the axis - need this??
        else:
            # image.shape = (150, 130, 1)
"""
