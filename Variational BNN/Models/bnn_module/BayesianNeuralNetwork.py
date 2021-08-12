import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tfk.layers
from . import activation_functions as af
from .CustomKerasModel import CustomKerasModel
from .DenseVariationalFlipout import DenseVariationalFlipout
from .callbacks import EpochDots

#
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class BayesianNeuralNetwork:
    def __init__(self, layers, pmf, prior, kl_loss_weight=1, activation='sigmoid', use_flipout=False, soft_coef_ll=1, alpha=None,
                 annealing_type='None',
                 kappa=None, start_annealing=None, end_lin_annealing=None, max_kl_weight=None, periods=None, batch_num=None, seed=42,
                 dtype='float64'):
        if layers is None:
            self.layers = [1, 20, 20, 1]
        else:
            self.layers = layers
        self.pmf = pmf
        self.prior = prior
        self.annealing_type = annealing_type
        self.kl_loss_weight = kl_loss_weight
        self.kappa = kappa
        self.start_annealing = start_annealing
        self.end_lin_annealing = end_lin_annealing
        self.max_kl_weight = max_kl_weight
        self.periods = periods
        self.batch_num = batch_num
        self.soft_coef_ll = soft_coef_ll

        if activation.lower() == 'LeakyRelu'.lower():
            # default 0.01
            if alpha is None:
                alpha = 0.15
            self.activation = tfkl.LeakyReLU(alpha=alpha)
        elif activation.lower() == 'ELU'.lower():
            # default 1.0
            if alpha is None:
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

        self.use_flipout = use_flipout
        self.seed = seed
        self.dtype = dtype
        tf.keras.backend.set_floatx(self.dtype)

        self.model = None
        self.lay_i = None
        self.y_pred, self.y_pred_samples, self.y_pred_sigma_samples, self.y_pred_var_samples = None, None, None, None
        self.var_mix_norm = None

        self.custom_model = True

    def build_model(self, custom_model=False, use_mean=False):
        self.custom_model = custom_model
        prior = self.prior.get_distribution
        posterior_mean_field = self.pmf.get_distribution

        if self.use_flipout:
            VarLayer = self.__dense_flipout
        elif not self.use_flipout:
            VarLayer = self.__dense_variational

        inputs = tf.keras.Input(shape=(self.layers[0],), name="input_layer", dtype=self.dtype)
        x = inputs
        for lay_i, nodes in enumerate(self.layers[1:-1]):
            self.__has_layer(lay_i)
            x = VarLayer(nodes, posterior_mean_field, prior, activation=self.activation)(x)
        self.__has_layer(len(self.layers) - 2)
        x = VarLayer(
            2 * self.layers[-1], posterior_mean_field, prior,
            activation=None)(x)
        outputs = tfp.layers.DistributionLambda(name='output_layer',
                                                make_distribution_fn=lambda t: tfd.Normal(loc=t[..., :self.layers[-1]],
                                                                                          scale=10 ** -5 + self.soft_coef_ll*tf.math.softplus(
                                                                                              t[...,
                                                                                              self.layers[-1]:])))(x)
        if self.custom_model:
            self.model = CustomKerasModel(inputs=inputs, outputs=outputs, annealing_type=self.annealing_type,
                                          kl_weight=self.kl_loss_weight,
                                          kappa=self.kappa, start_annealing=self.start_annealing, end_lin_annealing=self.end_lin_annealing, periods=self.periods,
                                          batch_num=self.batch_num,
                                          max_kl_weight=self.max_kl_weight, name='BNN_custom', use_mean=use_mean,
                                          dtype=self.dtype)
            # self.model.add_metric(tf.reduce_sum(self.model.losses), name="kl_loss")
            # self.model.add_metric(self.model.negloglik, name="negloglik")
        else:
            self.model = tfk.Model(inputs=inputs, outputs=outputs, name="BNN")

    def __dense_flipout(self, nodes, pmf, prior, activation):
        # return tfpl.DenseFlipout(nodes, kernel_posterior_fn=tfpl.util.default_mean_field_normal_fn(), kernel_prior_fn=tfpl.default_multivariate_normal_fn, activation=activation,
        #                          dtype=self.dtype)
        return tfpl.DenseFlipout(nodes, activation=activation,
                                 dtype=self.dtype)

        # return DenseVariationalFlipout(nodes, pmf, prior,
        #                                kl_weight=tf.constant(1, dtype=self.dtype), activation=activation,
        #                                dtype=self.dtype)

    def __dense_variational(self, nodes, pmf, prior, activation):
        return tfpl.DenseVariational(nodes, pmf, prior,
                                     kl_weight=tf.constant(1, dtype=self.dtype), activation=activation,
                                     dtype=self.dtype, kl_use_exact=False)

    def fit(self, train_data, val_data=None, epochs=10000, patience=1000, verbose=0, monitor='val_elbo'):

        epoch_dots = EpochDots(report_every=100, dot_every=1)

        if val_data is None:
            print("\n\n***Bayesian Neural Network***")
            print("Running for %s epochs" % epochs)
            history = self.model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=verbose,
                                     callbacks=[epoch_dots], shuffle=True)
            hist = history.history
            hist['epoch'] = history.epoch
        else:
            print("\n***Bayesian Neural Network***")
            print("Number of epochs: %s, with patience: %s" % (epochs, patience))
            early_stop = tfk.callbacks.EarlyStopping(monitor=monitor, mode='min', patience=patience,
                                                     restore_best_weights=True)
            history = self.model.fit(train_data, epochs=epochs, verbose=verbose,
                                     validation_data=val_data,
                                     callbacks=[epoch_dots, early_stop], shuffle=True)
            hist = history.history
            hist['epoch'] = history.epoch
            stopped_epoch = early_stop.stopped_epoch
            if stopped_epoch == 0:
                stopped_epoch = epochs
            hist['stopped_epoch'] = stopped_epoch

        return hist

    def compile(self, optimizer, loss, metrics=None, custom_metrics=None):
        if self.custom_model:
            self.model.special_compile(optimizer=optimizer, loss=loss, metrics=metrics, custom_metrics=custom_metrics)
        else:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    ######################### Prediction ############################

    def predict_dist(self, X):
        return self.model(X)

    # Currently just for target len = 1
    @tf.function
    def predict_singles(self, X, sample_dist_size=100, sample_size=100):
        all_samples = tf.TensorArray(
            tf.float64, size=0, dynamic_size=True, element_shape=[sample_size])
        for i in tf.range(sample_dist_size):
            n = self.model(X)
            s = tfd.Sample(
                n,
                sample_shape=sample_size)
            sample = tf.squeeze(s.sample())
            all_samples = all_samples.write(i, sample)
        return all_samples.stack()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float64, name="data"),
                                  tf.TensorSpec(shape=[], dtype=tf.int32, name="samples"),
                                  tf.TensorSpec(shape=[], dtype=tf.bool, name="bool")])
    def predict(self, X, sample_size, mean=False):
        self.y_pred_samples = tf.TensorArray(
            dtype=tf.float64, size=sample_size, dynamic_size=True,
            element_shape=[X.get_shape()[0], self.layers[-1]])
        self.y_pred_var_samples = tf.TensorArray(
            dtype=tf.float64, size=sample_size, dynamic_size=True,
            element_shape=[X.get_shape()[0], self.layers[-1]])

        if mean:
            for i in tf.range(sample_size):
                y_dist = self.model(X)
                y_pred_i = y_dist.mean()
                # y_pred_i = self.model.predict(X)
                y_var_i = y_dist.variance()
                self.y_pred_samples = self.y_pred_samples.write(i, y_pred_i)
                self.y_pred_var_samples = self.y_pred_var_samples.write(i, y_var_i)
        elif not mean:
            for i in tf.range(sample_size):
                y_dist = self.model(X)
                y_var_i = y_dist.variance()
                self.y_pred_samples = self.y_pred_samples.write(i, y_dist)
                self.y_pred_var_samples = self.y_pred_var_samples.write(i, y_var_i)

        return tf.squeeze(self.y_pred_samples.stack()), tf.squeeze(self.y_pred_var_samples.stack())

    @classmethod
    def asym_interval(cls, mus, sigma, n, sign):
        mu = mus.mean(axis=0)
        # mu: predictions
        # sigma: standard deviations
        # n: number of credible intervals, analog to number std.
        # sign: Upper or lower interval(upper +, lower -)
        s = sign
        y = np.exp(mu)
        delta = (-y + np.exp(mu + s * sigma))
        M = (y / delta) * ((1 + s * delta / y) ** n - 1)
        interval = M * delta
        return interval

    @classmethod
    def asym_sigma(cls, mus, sigma, n, sign):
        mu = mus.mean(axis=0)
        # mu: predictions
        # sigma: standard deviations
        # n: number of credible intervals, analog to number std.
        # sign: Upper or lower interval(upper +, lower -)
        s = sign
        y = np.exp(mu)
        delta = (-y + np.exp(mu + s * sigma))
        M = (y / delta) * ((1 + s * delta / y) ** n - 1)
        interval = y + s * M * delta
        return interval

    def predict_sigmas(self, y_pred_samples=None, y_pred_var_samples=None, inv=True, n=1):
        # Fix it such that you only use var not sigma
        if not any(a is None for a in [y_pred_samples, y_pred_var_samples]):
            self.y_pred_samples, self.y_pred_var_samples = y_pred_samples, y_pred_var_samples
            var_aleo = self.y_pred_var_samples.mean(axis=0)

            var_epis = (self.y_pred_samples ** 2).mean(axis=0) - self.y_pred_samples.mean(axis=0) ** 2

            var_mix = var_aleo + var_epis
            self.var_mix_norm = np.copy(var_mix)

            if not inv:
                return np.sqrt(var_aleo), np.sqrt(var_epis), np.sqrt(var_mix)
            elif inv:
                y = self.y_pred_samples
                sigma_aleo_ls = np.vstack((self.asym_interval(y, np.sqrt(var_aleo), n, sign=-1),
                                           self.asym_interval(y, np.sqrt(var_aleo), n, sign=+1)))
                sigma_epis_ls = np.vstack((self.asym_interval(y, np.sqrt(var_epis), n, sign=-1),
                                           self.asym_interval(y, np.sqrt(var_epis), n, sign=+1)))
                sigma_mix_ls = np.vstack((self.asym_interval(y, np.sqrt(var_mix), n, sign=-1),
                                          self.asym_interval(y, np.sqrt(var_mix), n, sign=+1)))
                return sigma_aleo_ls, sigma_epis_ls, sigma_mix_ls

    def __has_layer(self, lay_i):
        if hasattr(self.pmf, 'layer_i'):
            self.pmf.layer_i = lay_i
        if hasattr(self.prior, 'layer_i'):
            self.prior.layer_i = lay_i
