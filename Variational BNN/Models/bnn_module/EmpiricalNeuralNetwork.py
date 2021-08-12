import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from . import activation_functions as af
from .callbacks import EpochDots
import pickle

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tfk.layers


class PriorNeuralNetwork:
    # main class for BNN
    def __init__(self, layers_ls=None, activation='relu', dtype='float64'):
        self.dtype = dtype
        tf.keras.backend.set_floatx(self.dtype)
        if layers_ls is None:
            print("Please set the layers architecture")
            return
        else:
            self.layers_ls = layers_ls
        if activation.lower() == 'LeakyRelu'.lower():
            alpha = 0.15
            self.activation = tfkl.LeakyReLU(alpha=alpha)
        elif activation.lower() == 'ELU'.lower():
            alpha = 1
            self.activation = tfkl.ELU(alpha=alpha)
        elif activation.lower() == 'SELU'.lower():
            self.activation = af.selu
        elif activation.lower() == 'swish'.lower():
            self.activation = af.swish
        elif activation.lower() == 'mish'.lower():
            self.activation = af.mish
        else:
            self.activation = activation

        self.model = None
        self.best_loss = None
        self.best_rmse = None
        self.last_epoch = None
        self.best_epoch = None
        self.history = None
        self.soft0_output = 0.01,
        self.soft1_output = 1e-5,

    @classmethod
    def __negloglik(cls, y, rv_y):
        nloglik = -rv_y.log_prob(y)
        return nloglik

    def build_model(self):
        inputs = tf.keras.Input(shape=(self.layers_ls[0],), name="input_layer", dtype=self.dtype)
        x = inputs
        for nodes in self.layers_ls[1:-1]:
            x = tfkl.Dense(nodes, activation=self.activation, dtype=self.dtype)(x)
            # x = layers.BatchNormalization()(x)
        x = tfkl.Dense(2 * self.layers_ls[-1], activation=None, dtype=self.dtype)(x)
        output_layer = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Normal(loc=t[..., :self.layers_ls[-1]],
                                                      scale=10 ** -5 + tf.math.softplus(t[..., self.layers_ls[-1]:])))(
            x)
        self.model = tf.keras.Model(inputs=inputs, outputs=output_layer, name="Empirical_NN")

    def compile(self, optimizer_type, metrics=None):
        if metrics is None:
            metrics = ['mse']
        self.model.compile(optimizer=optimizer_type, loss=self.__negloglik, metrics=metrics)

    def fit(self, train_data, val_data=None, epochs=10000, patience=1000):

        epoch_dots = EpochDots(report_every=100, dot_every=1)
        print("\n***Empirical Neural Network***")
        if val_data is None:
            # Fix
            # epochs = 10000
            print("Running for %s epochs" % epochs)
            history = self.model.fit(train_data, epochs=epochs, verbose=0,
                                     callbacks=[epoch_dots])
            hist_df = pd.DataFrame(history.history)
            hist_df['epoch'] = history.epoch

        else:
            print("Number of epochs: %s, with patience: %s" % (epochs, patience))
            early_stop = tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience,
                                                     restore_best_weights=True)
            history = self.model.fit(train_data, epochs=epochs, verbose=0,
                                     callbacks=[epoch_dots, early_stop], validation_data=val_data)
            print("\nBest val loss: ", early_stop.best)
            self.best_loss = early_stop.best
            self.last_epoch = early_stop.stopped_epoch
            if self.last_epoch == 0:
                self.last_epoch = epochs
            # hist_df = pd.DataFrame(history.history)
            hist = history.history
            hist['epoch'] = history.epoch
            self.best_epoch = np.argmin(hist['val_loss']) + 1
            self.history = hist
            self.best_rmse = min(history.history['val_rmse'])
            print("Best val rmse: ", self.best_rmse)
            print("Best epoch: ", self.best_epoch)

        return self.model, self.history

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_dist(self, X):
        return self.model(X)

    def get_weights(self):
        if self.model is not None:
            weights_ls = []
            for layer in self.model.layers[1:-1]:
                layer = layer.get_weights()
                w, b = layer[0], layer[1]
                params = np.append(w.ravel(), b.ravel())
                weights_ls.append(params)
            return weights_ls
        else:
            print("There's no self.model")

    def summary(self):
        return self.model.summary()
