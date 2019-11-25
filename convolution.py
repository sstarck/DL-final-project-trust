import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):

        """
        The Model class predicts the attribute scores for a batch of images.
        """

        super(Model, self).__init__()

        # initialize model parameters
        # hyperparameters passed to this part?

        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        # experiment with optimizers

        # initialize layers

        self.conv1_1 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), padding='same', activation=tf.keras.layers.ReLU, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), input_shape=(self.batch_size, 150, 130))
        self.conv1_2 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), padding='same', activation=tf.keras.layers.ReLU, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None))
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        # add batch norm?
        # add relu?
        self.conv2_1 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), padding='same', activation=tf.keras.layers.ReLU, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None))
        self.conv2_2 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), padding='same', activation=tf.keras.layers.ReLU, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None))
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        # add batch norm?
        # add relu?
        self.conv3_1 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), padding='same', activation=tf.keras.layers.ReLU, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None))
        self.conv3_2 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), padding='same', activation=tf.keras.layers.ReLU, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None))
        self.conv3_3 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(2, 2), padding='same', activation=tf.keras.layers.ReLU, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None))
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        # add batch norm?
        # add relu?
        self.dropout1 = tf.keras.layers.Dropout(rate=0.3)
        # experiment with dropout rate
        self.relu1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.3)
        # experiment with dropout rate
        self.relu2 = tf.keras.layers.Dense(128, activation='relu')
        # how many relus before final dense?
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """
        :param inputs: images of shape (batch_size, 150, 130, 3)
        :return: the batch attribute scores as a tensor
        """

        conv1_output = self.conv1_2( self.conv1_1(inputs) )
        pool1_output = self.pool1(conv1_output)

        conv2_output = self.conv2_2( self.conv2_1(pool1_output) )
        pool2_output = self.pool2(conv2_output)

        conv3_output = self.conv3_3( self.conv3_2( self.conv3_1(pool2_output) ) )
        pool3_output = self.pool3(conv3_output)

        flatten = tf.reshape(pool3_output, [inputs.shape[0], -1])
        relu1_output = self.relu1( self.dropout1(flatten) )
        relu2_output = self.relu2( self.dropout2(relu1_output) )
        predicted_scores = self.dense(relu2_output)

        return predicted_scores

    def loss(self, predicted_scores, actual_scores):
        """
        Calculates loss of the predictions for this batch

        :param predicted_scores: a matrix of shape (batch_size, 1) as a tensor containing the model's predicted scores
        :param actual_scores: matrix of shape (batch_size, 1) containing the actual scores
        :return: the loss of the model as a tensor of size 1
        """
        # use MAE or R^2 instead of (in addition to) MSE?
        return tf.keras.losses.MSE(actual_scores, predicted_scores)

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

    for i in range(num_iterations):

        batch_inputs = train_inputs[n:n+batch_size, :, :]
        batch_scores = train_labels[n:n+batch_size]
        n += batch_size

        with tf.GradientTape() as tape:
            predicted_scores = model.call(batch_inputs)
            loss = model.loss(batch_scores, predicted_scores)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_images, 150, 130, 3)
    :param test_labels: train labels (all labels for testing) of shape (num_images,)
    :returns: total loss across all batches

    """
    n = 0
    loss = 0
    batch_size = model.batch_size
    num_iterations = int( np.size(test_inputs, 0) / batch_size )

    for i in range(num_iterations):

        batch_inputs = test_inputs[n:n+batch_size, :, :]
        batch_scores = test_labels[n:n+batch_size]
        n += batch_size

        predicted_scores = model.call(batch_inputs)
        loss += model.loss(batch_scores, predicted_scores)
				# R^2 = 1 - mse/var
        # is it okay to add mse's for this calculation?
        # need to add variance in order to calculate R^2 --> how to

    return loss
