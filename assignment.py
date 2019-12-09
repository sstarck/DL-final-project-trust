#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import preprocess
import convolution
import numpy as np
import tensorflow as tf
import json
import random
import matplotlib.pyplot as plt
import operator
import time

def main():
    print("Running preprocessing...")
    # Load in images, labels, and hyperparameters
    ATTRIBUTE = 'Trustworthiness' # Could change to dominance, IQ, etc.
    LABELS = 'Annotations/' + ATTRIBUTE + '/annotations.csv'
    SPACE = 'Spaces/' + ATTRIBUTE + '_space.json'
    train_data, train_labels = preprocess.get_data(directory='Images/Train/', path=LABELS, attribute=ATTRIBUTE)
    test_data, test_labels = preprocess.get_data(directory='Images/Test/', path=LABELS, attribute=ATTRIBUTE)
    with open(SPACE, 'r') as f: # Load hyperparameter file
        hyperparameters = json.load(f)
    # Add noise, brightness, and flip a third of training images to preprocess
    third = len(train_data) // 3
    train_data[:third] = preprocess.random_noise(train_data[:third], hyperparameters)
    train_data[third:(third*2)] = preprocess.brightness(train_data[third:(third*2)], hyperparameters)
    train_data[(third*2):] = preprocess.flip(train_data[(third*2):])
    print("Preprocessing complete.")
    model = convolution.Model() # Initialize model
    print("Training model ...")
    num_epochs = 10
    r2_data = []
    loss_data = []
    for i in range(num_epochs): # Shuffle data before training each epoch
        t0 = time.time()
        indices = list(range(0, len(train_labels)))
        tf.random.shuffle(indices)
        shuffled_train_data = tf.gather(train_data, indices)
        shuffled_train_labels = tf.gather(train_labels, indices)
        epoch_train_loss = convolution.train(model, shuffled_train_data, shuffled_train_labels)
        loss_data.append( epoch_train_loss )
        r2, all_predicted_scores = convolution.test(model, test_data, test_labels)
        r2_data.append(r2)
        t = time.time() - t0
        mins = int( (t/60) % 60 )
        print('')
        print('Epoch', i+1, 'done after', mins, 'minutes and', int( t-60*mins ), 'seconds')
        print('Average loss for epoch', i+1, 'was', epoch_train_loss)
        print('R-Squared for epoch', i+1, 'was', r2)
        print('')

    indices = range(1, len(train_labels)+1)
    all_predicted_scores = [x[0] for x in all_predicted_scores.numpy().tolist()]
    zipped_images = zip(all_predicted_scores, indices)
    sorted_images = sorted(zipped_images, key = operator.itemgetter(0))
    print(sorted_images)
    epochs = range(1, num_epochs+1)

    fig1 = plt.figure(1)
    plt.plot(epochs, r2_data)
    plt.xlabel("Epoch")
    plt.ylabel("R-Squared")

    fig2 = plt.figure(2)
    plt.plot(epochs, loss_data)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")

    plt.show()

if __name__ == '__main__':
    main()
