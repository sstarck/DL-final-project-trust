#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats.mstats import linregress
import matplotlib.pyplot as plt

class Model(tf.keras.Model):
    def __init__(self):

        """
        The Model class predicts the attribute scores for a batch of images.
        """

        super(Model, self).__init__()

        # initialize model parameters

        self.batch_size = 16
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=10**-4.23396310557871, rho=0.9, epsilon=1e-8)
        self.epsilon = 0.001

        # initialize filters and fully connected layers

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        
        self.conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv6 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        
        self.conv7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv8 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv9 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        
        self.conv10 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv11 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv12 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        
        self.conv13 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv14 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.conv15 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', activation='relu')
        self.pool6 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.5478380034535812)
        self.relu = tf.keras.layers.Dense(2079, activation='relu')
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, dropout):
        """
        :param inputs: images of shape (batch_size, 150, 130, 3)
        :param dropout: include dropout layers in training but not in testing
        :return: the batch attribute scores as a tensor
        """

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        pool1 = self.pool1(conv2)

        conv3 = self.conv3(pool1)
        conv4 = self.conv4(conv3)
        pool2 = self.pool2(conv4)

        conv5 = self.conv5(pool2)
        conv6 = self.conv6(conv5)
        pool3 = self.pool3(conv6)

        conv7 = self.conv7(pool3)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)
        pool4 = self.pool4(conv9)

        conv10 = self.conv10(pool4)
        conv11 = self.conv11(conv10)
        conv12 = self.conv12(conv11)
        pool5 = self.pool5(conv12)

        conv13 = self.conv13(pool5)
        conv14 = self.conv14(conv13)
        conv15 = self.conv15(conv14)
        pool6 = self.pool6(conv15)

        flatten = self.flatten(pool6)
        relu = self.relu( self.dropout(flatten) )
        predicted_scores = self.dense(relu)

        return predicted_scores

    def loss(self, actual_scores, predicted_scores):
        """
        Calculates loss of the predictions for this batch

        :param predicted_scores: a matrix of shape (batch_size, 1) as a tensor containing the model's predicted scores
        :param actual_scores: matrix of shape (batch_size, 1) containing the actual scores
        :return: the mean squared error of the batch as a tensor of size 1
        """
        return tf.reduce_mean( tf.keras.losses.MSE(tf.reshape(actual_scores, [-1, 1]), tf.reshape(predicted_scores, [-1, 1])) )

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_images, 150, 130, 3)
    :param train_labels: train labels (all labels for training) of shape (num_images,)
    :return: None
    """
    n = 0
    batch_size = model.batch_size
    num_iterations = int( np.size(train_inputs, 0) / batch_size )
    big_loss = []
    for i in range(num_iterations):
        
        batch_inputs = train_inputs[n:n+batch_size, :, :, :]
        batch_scores = train_labels[n:n+batch_size,]
        n += batch_size

        with tf.GradientTape() as tape:
            predicted_scores = model.call(batch_inputs, True)
            loss = model.loss(batch_scores, predicted_scores)

        print("loss for batch", i + 1, '/', num_iterations, '=', loss)
        big_loss.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return tf.reduce_mean(big_loss)

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_images, 150, 130, 3)
    :param test_labels: train labels (all labels for testing) of shape (num_images,)
    :returns: average R^2 across all batches

    """
    n = 0
    R_squared = 0
    batch_size = model.batch_size
    num_iterations = int( np.size(test_inputs, 0) / batch_size )
    all_predicted_scores=[]

    for i in range(num_iterations):
        batch_inputs = test_inputs[n:n+batch_size, :, :, :]
        batch_scores = test_labels[n:n+batch_size,]
        n += batch_size

        predicted_scores = model.call(batch_inputs, False)
        m, b, r, p, e = linregress(y=tf.reshape(batch_scores, [-1, 1]), x=tf.reshape(predicted_scores, [-1, 1]))
        r2 = r**2
        
        if i == 0:
            all_predicted_scores = predicted_scores
        else:
            all_predicted_scores = tf.concat([all_predicted_scores, predicted_scores], axis=0)

        R_squared += r2

    return R_squared/num_iterations, all_predicted_scores
