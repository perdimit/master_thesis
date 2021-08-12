import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .EmpiricalNeuralNetwork import PriorNeuralNetwork as PNN
# from sklearn.metrics import mean_squared_error as mse
from . import error_handling as eh
import time
import datetime

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers


class SetDistribution:
    """
    This class provides a set distribution method for multiple Bayesian neural network(BNN) runs.
    The BNN initializer will check which initializer_type has been set and provide the right set of parameters
    from the dict of parameters that was passed into BNN initialization. The BNN variational layers then fetches
    the prior/posterior by using 'get distribution'

    In this class, and in inherited classes, mu and sigma refers to the mean and the standard deviation of the initializing distribution,
    while loc and scale is the mean and the input to the softplus function that gives the standard deviation of the final distribution.
    So sigma_loc means the standard deviation(sigma) of the distribution in which the mean(loc) of the final distribution is sampled from.
    """

    def __init__(self, dtype="float64", trainable=True, dist='Guassian'):
        self.dtype = dtype
        self.trainable = trainable
        self.c = np.log(np.expm1(1.))
        self.soft_coef = 1
        self.dist = dist
        self.initializer_type = self._constant_initializer

    # Arguments of set functions must have unique names relative to priors.py functions
    def set_constant_initializer(self, loc=0, scale=1, soft_coef=1):
        self.initializer_type = self._constant_initializer
        self.loc = loc
        self.scale = np.log(np.exp(scale) - 1) - self.c
        self.soft_coef = soft_coef

    def set_random_gaussian_initializer(self, loc=0, scale=0.05, sigma_loc=0.1, sigma_scale=0.1, soft_coef=1):
        # Setting the means and the standard deviations of the two normal distributions that loc(prior mean) and scale(prior stddev) is sampled from.
        # the scale sample is passed into a softplus function.
        # scale: the mean of standard deviations
        self.initializer_type = self._random_gaussian_initializer
        self.loc = loc
        self.sigma_loc = sigma_loc
        self.scale = np.log(np.exp(scale) - 1)
        self.sigma_scale = sigma_scale
        self.soft_coef = soft_coef

    def _constant_initializer(self, shape, dtype=None):
        n = int(shape / 2)
        loc = tf.cast(tf.fill((n,), self.loc), dtype)
        scale = tf.cast(tf.fill((n,), self.scale), dtype)
        return tf.concat([loc, scale], 0)

    def _random_gaussian_initializer(self, shape, dtype):
        n = int(shape / 2)
        loc_norm = tf.random_normal_initializer(mean=self.loc, stddev=self.sigma_loc)
        loc = tf.Variable(
            initial_value=loc_norm(shape=(n,), dtype=dtype),
            trainable=self.trainable,
        )
        scale_norm = tf.random_normal_initializer(mean=self.scale, stddev=self.sigma_scale)
        scale = tf.Variable(
            initial_value=scale_norm(shape=(n,), dtype=dtype),
            trainable=self.trainable,
        )
        return tf.concat([loc, scale], 0)

    def get_distribution(self, kernel_size, bias_size=0, dtype=None):
        if dtype is None:
            dtype = self.dtype
        n = kernel_size + bias_size
        if self.dist == 'Laplace':
            return tfk.Sequential([
                tfpl.VariableLayer(2 * n, dtype=dtype,
                                   initializer=lambda shape, dtype: self.initializer_type(shape, dtype),
                                   trainable=self.trainable),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Laplace(loc=t[..., :n],
                                scale=1e-5 + self.soft_coef * tf.nn.softplus(self.c + t[..., n:])),
                    reinterpreted_batch_ndims=1))
            ])
        else:
            return tfk.Sequential([
                tfpl.VariableLayer(2 * n, dtype=dtype,
                                   initializer=lambda shape, dtype: self.initializer_type(shape, dtype),
                                   trainable=self.trainable),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               scale=1e-5 + self.soft_coef * tf.nn.softplus(self.c + t[..., n:])),
                    reinterpreted_batch_ndims=1))
            ])


# class SetMixtureDistribution:
#     def __init__(self, dtype="float64", trainable=False):
#         self.dtype = dtype
#         self.trainable = trainable
#         self.c = tf.cast(np.log(np.expm1(1.)), dtype=self.dtype)
#         self.soft_coef = 1
#         self.initializer_type = self._constant_mixture_initializer
#
#     def set_constant_mixture_initializer(self, loc1=0, scale1=0.1, loc2=0, scale2=1, pi=0.5, soft_coef=1):
#         self.loc1 = tf.cast(loc1, dtype=self.dtype)
#         self.loc2 = tf.cast(loc2, dtype=self.dtype)
#         self.scale1 = tf.cast(np.log(np.exp(scale1) - 1) - self.c, dtype=self.dtype)
#         self.scale2 = tf.cast(np.log(np.exp(scale2) - 1) - self.c, dtype=self.dtype)
#         self.pi = tf.cast(pi, dtype=self.dtype)
#         self.soft_coef = tf.cast(soft_coef, dtype=self.dtype)
#         self.initializer_type = self._constant_mixture_initializer
#
#     def _constant_mixture_initializer(self, shape, dtype=None):
#         n = int(shape / 4)
#         # print(dtype)
#         # pi = tf.cast(tf.fill((n,), self.pi), dtype)
#         loc1 = tf.cast(tf.fill((n,), self.loc1), dtype)
#         loc2 = tf.cast(tf.fill((n,), self.loc2), dtype)
#         scale1 = tf.cast(tf.fill((n,), self.scale1), dtype)
#         scale2 = tf.cast(tf.fill((n,), self.scale2), dtype)
#         return tf.concat([loc1, loc2, scale1, scale2], 0)
#         # return tf.zeros(shape)
#
#     def get_distribution(self, kernel_size, bias_size=0, dtype=None):
#         if dtype is None:
#             dtype = self.dtype
#         n = kernel_size + bias_size
#         return tf.keras.Sequential([
#             tfpl.VariableLayer(4 * n, dtype=dtype,
#                                initializer=lambda shape, dtype: self.initializer_type(shape, dtype),
#                                trainable=self.trainable),
#             tfp.layers.DistributionLambda(lambda t:
#                                           tfd.MixtureSameFamily(
#                                               mixture_distribution=tfd.Categorical(probs=[self.pi, 1 - self.pi],
#                                                                                    dtype='int32')
#                                               , components_distribution=tfd.Independent(
#                                                   tfd.Normal(loc=[t[..., :n], t[..., n:2 * n]],
#                                                              scale=[1e-5 + tf.nn.softplus(self.c + t[..., 2 * n:3 * n]),
#                                                                     1e-5 + tf.nn.softplus(self.c + t[..., 3 * n:4 * n])]
#                                                              ), reinterpreted_batch_ndims=1)
#                                               ))])


class SetNNDistribution:
    def __init__(self, dtype="float64", trainable=True, link_object=None, strategy=None, dist='Gaussian'):
        self.dtype = dtype
        self.trainable = trainable
        self.params = None
        self.mle = None
        if link_object is not None:
            if not hasattr(link_object, 'mle'):
                print(
                    "Link object has no attribute mle, i.e. no network weights.")
        self.strategy = strategy
        self.link_object = link_object
        self.c = np.log(np.expm1(1.))
        self.layer_i = None
        self.soft_coef = 1
        self.dist = dist
        self.initializer_type = self._nn_complex_initializer
        self.best_epoch = None
        self.epochs = None
        self.time_used = None
        self.history = None

    def set_nn_constant_initializer(self, scale=1, soft_coef=1, train_data=None, val_data=None, layers=None,
                                    epochs=10000,
                                    patience=500,
                                    optimizer_type='Adam',
                                    learning_rate=0.01, activation='relu', batch_size=32):
        self.initializer_type = self._nn_constant_initializer
        self.scale = np.log(np.exp(scale) - 1) - self.c
        self.soft_coef = soft_coef
        if None in {train_data}:
            if self.link_object is not None:
                self.mle = self.link_object.mle
        else:
            self._set_neural_network(train_data, val_data, layers, epochs, patience, optimizer_type, learning_rate,
                                     activation, batch_size)

    def set_nn_gaussian_initializer(self, mu_scale=-3, sigma_scale=0.1, soft_coef=1, train_data=None, val_data=None,
                                    layers=None,
                                    epochs=10000, patience=500, optimizer_type='Adam',
                                    learning_rate=0.01, activation='relu', batch_size=32):
        self.initializer_type = self._nn_gaussian_initializer
        self.sigma_scale = tf.cast(sigma_scale, dtype=self.dtype)
        self.mu_scale = tf.cast(mu_scale, dtype=self.dtype)
        self.soft_coef = soft_coef
        if None in {train_data}:
            if self.link_object is not None:
                self.mle = self.link_object.mle
        else:
            self._set_neural_network(train_data, val_data, layers, epochs, patience, optimizer_type, learning_rate,
                                     activation, batch_size)

    def set_nn_complex_initializer(self, delta=1e-06, soft_coef=1, train_data=None, val_data=None, layers=None,
                                   epochs=10000, patience=500,
                                   optimizer_type='Adam', learning_rate=0.01, activation='relu', batch_size=32):
        self.initializer_type = self._nn_complex_initializer
        self.delta = tf.cast(delta, dtype=self.dtype)
        self.soft_coef = soft_coef
        if None in {train_data}:
            if self.link_object is not None:
                self.mle = self.link_object.mle
        else:
            self._set_neural_network(train_data, val_data, layers, epochs, patience, optimizer_type, learning_rate,
                                     activation, batch_size)

    def _set_neural_network(self, train_data, val_data, *args):
        """
        This method checks if the arguments passed in(except train_data, val_data, delta and sigma) are the same as in the previous network run in the same instance of this class.
        So if delta and sigma are being changed, all else equal, there's no need to run the network again.

        This method must be included for Initialization.py to recognize the object as containing a neural network.
        """
        run_network = True
        if self.params is not None:
            i = 0
            while True:
                if args[i] not in self.params:
                    break
                elif i == len(args) - 1:
                    run_network = False
                    break
                i += 1

        if run_network:
            self.params = args
            self.train_data = train_data
            self.val_data = val_data
            self.__train_neural_network(train_data, val_data)

    def _nn_constant_initializer(self, shape, dtype=None):
        n = int(shape / 2)
        scale = tf.cast(tf.fill((n,), self.scale), dtype)
        return tf.concat([self.mle[self.layer_i], scale], 0)

    def _nn_gaussian_initializer(self, shape, dtype=None):
        """
        The _nn_gaussian_initializer sets the means of the prior/posterior mean field equal to optimal weights(MLE) of the neural network,
        but allows for manual initialization of the normal distribution that the standard deviation(scale) is sampled from.
        """
        n = int(shape / 2)
        if self.mle is not None:
            loc = tf.cast(self.mle[self.layer_i], dtype=dtype)
        else:
            print("DistributionInitialization: please initialize the neural network prior first. Check if link_object is properly set in prior/pmf")
            return
        scale = K.random_normal((n,), mean=self.mu_scale, stddev=self.sigma_scale, dtype=dtype)
        return tf.concat([loc, scale], 0)

    def _nn_complex_initializer(self, shape, dtype=None):
        """
        The _nn_complex_initializer sets the means of the prior/posterior mean field equal to optimal weights(MLE) of the neural network,
        and sets the standard deviation, rho, of the final distribution also as a function of the neural network weights, but with a 
        hyperparameter, delta, that must be tuned.
        """
        n = int(shape / 2)
        if self.mle is not None:
            loc = tf.cast(self.mle[self.layer_i], dtype=dtype)
        else:
            print("DistributionInitialization: self.mle is None, please initialize the neural network prior first. \n "
                  "Check if link_object is properly set in prior/pmf. Linked object must be passed in together [prior, pmf], and it must be linked to a unique object not use by other objects.")
            return
        delta_loc = self.delta * np.abs(loc)
        rho = np.log(np.expm1(delta_loc) + 1e-25)
        # print("Weights: ", tf.concat([loc, rho], 0))
        return tf.concat([loc, rho], 0)

    def __train_neural_network(self, train_data, val_data=None):
        layers, epochs, patience, optimizer, activation = self.params[0], self.params[1], self.params[2], eh.optimizer(
            self.params[3], self.params[4]), self.params[5]
        pnn = PNN(layers_ls=layers, activation=activation, dtype=self.dtype)
        if self.strategy is not None:
            with self.strategy.scope():
                pnn.build_model()
                pnn.compile(optimizer_type=optimizer, metrics=[tfk.metrics.RootMeanSquaredError(name='rmse')])
        else:
            pnn.build_model()
            pnn.compile(optimizer_type=optimizer, metrics=[tfk.metrics.RootMeanSquaredError(name='rmse')])

        # pnn.summary()
        t = time.time()
        if val_data is None:
            if self.epochs is not None:
                epochs = self.epochs
            _, self.history = pnn.fit(train_data, epochs=epochs)
        else:
            _, self.history = pnn.fit(train_data, val_data, epochs=epochs, patience=patience)
        run_time = time.time() - t
        self.time_used = str(datetime.timedelta(seconds=round(run_time)))
        self.mle = pnn.get_weights()
        self.best_epoch = pnn.best_epoch

    def get_distribution(self, kernel_size, bias_size=0, dtype=None):
        if dtype is None:
            dtype = self.dtype
        n = kernel_size + bias_size
        if self.dist == 'Laplace':
            return tfk.Sequential([
                tfpl.VariableLayer(2 * n, dtype=dtype,
                                   initializer=lambda shape, dtype: self.initializer_type(shape, dtype),
                                   trainable=self.trainable),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Laplace(loc=t[..., :n],
                                scale=1e-5 + self.soft_coef * tf.nn.softplus(self.c + t[..., n:])),
                    reinterpreted_batch_ndims=1))
            ])
        elif self.dist == 'Gaussian':
            return tfk.Sequential([
                tfpl.VariableLayer(2 * n, dtype=dtype,
                                   initializer=lambda shape, dtype: self.initializer_type(shape, dtype),
                                   trainable=self.trainable),
                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                    tfd.Normal(loc=t[..., :n],
                               scale=1e-5 + self.soft_coef * tf.nn.softplus(self.c + t[..., n:])),
                    reinterpreted_batch_ndims=1))
            ])
        else:
            print("DistributionInitialization: %s is not availabe. Use dist='Laplace' or 'Gaussian'" % self.dist)

