import preprocess
import convolution
import numpy as np
import tensorflow as tf
import json

def main():
    # TO DO: Get data and proprocess
    print("Running preprocessing...")
    # Load in images, labels, and hyperparameters
    ATTRIBUTE = 'Trustworthiness'
    LABELS = 'Annotations/' + ATTRIBUTE + '/annotations.csv'
    SPACE = 'Spaces/' + ATTRIBUTE + '_space.json'
    train_data, train_labels = preprocess.get_data(directory='Images/Train/', annotation=LABELS, attribute=ATTRIBUTE, norm=True)
    test_data, test_labels = preprocess.get_data(directory='Images/Test/', annotation=LABELS, attribute=ATTRIBUTE, norm=True)
    with open(SPACE, 'r') as f: # Load hyperparameter file
        hyperparameters = json.load(f)
    print("Preprocessing complete.")

    print("Running tests of image preprocessing...")
    image = train_data[0] # Get sample image (first in training data)
    # print("Image = ", image)
    noisy_image = preprocess.random_noise(image, hyperparameters)
    bright_image = preprocess.brightness(image, hyperparameters)
    flipped_image = preprocess.flip(image)
    print("Tests completed :)")

    # TO DO: ADD NOISE/ BRIGHTNESS/ FLIPPING TO TRAIN_IMAGES ONCE GET BASELINE!

    # TODO: Initialize model
    model = convolution.Model()
    # TODO: Train and test model
    print("Training model ...")
    convolution.train(model, train_data, train_labels)
    print("Training complete. Testing model ...")
    loss = convolution.test(model, test_data, test_labels)
    print("Testing complete. Loss = ", loss)

if __name__ == '__main__':
    main()
