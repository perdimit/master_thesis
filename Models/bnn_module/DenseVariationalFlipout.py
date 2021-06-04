# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DenseVariationalFlipout layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.math import random_rademacher as rademacher
from tensorflow_probability.python.util import SeedStream
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.layers.dense_variational_v2 import DenseVariational


# from tensorflow_probability.python.random import rademacher


class DenseVariationalFlipout(DenseVariational):
    """Dense layer with random `kernel` and `bias`.
    See for parent layer https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/dense_variational_v2.py#L26-L145

  This layer uses variational inference to fit a "surrogate" posterior to the
  distribution over both the `kernel` matrix and the `bias` terms which are
  otherwise used in a manner similar to `tf.keras.layers.Dense`.

  This layer fits the "weights posterior" according to the following generative
  process:

  ```none
  [K, b] ~ Prior()
  M = matmul(X, K) + b
  Y ~ Likelihood(M)
  ```

  """

    def __init__(self,
                 units,
                 make_posterior_fn,
                 make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=False,
                 activation=None,
                 use_bias=True,
                 seed=42,
                 activity_regularizer=None,
                 **kwargs):

        super(DenseVariationalFlipout, self).__init__(units=units,
                                                      make_posterior_fn=make_posterior_fn,
                                                      make_prior_fn=make_prior_fn,
                                                      kl_weight=kl_weight,
                                                      kl_use_exact=kl_use_exact,
                                                      activation=activation,
                                                      use_bias=use_bias,
                                                      activity_regularizer=activity_regularizer,
                                                      **kwargs)
        self.kl_weight = kl_weight
        self.seed = seed

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'make_posterior_fn': self.make_posterior_fn,
            'make_prior_fn': self.make_prior_fn,
            'kl_weight': self.kl_weight,
            'kl_use_exact': self.kl_use_exact,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'seed': self.seed,
            'activity_regularizer': self.activity_regularizer,
        })
        return config

    def call(self, inputs):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        inputs = tf.cast(inputs, dtype, name='inputs')
        q = self._posterior(inputs)
        r = self._prior(inputs)
        self.add_loss(self._kl_divergence_fn(q, r))

        w = tf.convert_to_tensor(value=q)

        prev_units = self.input_spec.axes[-1]
        if self.use_bias:
            split_sizes = [prev_units * self.units, self.units]
            kernel, bias = tf.split(w, split_sizes, axis=-1)
            kernel_std, bias_std = tf.split(q.stddev(), split_sizes, axis=-1)
        else:
            kernel, bias = w, None
            kernel_std = q.stddev()

        kernel = tf.reshape(kernel, shape=tf.concat([
            tf.shape(kernel)[:-1],
            [prev_units, self.units],
        ], axis=0))
        outputs = tf.matmul(inputs, kernel)

        # # Kernel affine
        input_shape = tf.shape(inputs)
        batch_shape = input_shape[:-1]
        seed_stream = SeedStream(self.seed, salt='Flipout')
        kernel_std = tf.reshape(kernel_std, shape=tf.concat([
            tf.shape(kernel_std)[:-1],
            [prev_units, self.units],
        ], axis=0))
        affine_kernel = normal_lib.Normal(loc=tf.zeros(shape=list(kernel.get_shape()), dtype=inputs.dtype), scale=kernel_std).sample()
        # tf.print("affine kernel: ", affine_kernel)
        # affine_kernel = tf.random.normal(shape=list(kernel.get_shape()), stddev=kernel_std)
        # tf.print("std: ", kernel_std)

        # tf.print(input_shape)
        sign_input = rademacher(
            input_shape,
            dtype=inputs.dtype,
            seed=seed_stream())
        sign_output = rademacher(
            tf.concat([batch_shape,
                       tf.expand_dims(self.units, 0)], 0),
            dtype=inputs.dtype,
            seed=seed_stream())
        perturbed_inputs = tf.matmul(
            inputs * sign_input, affine_kernel) * sign_output
        # tf.print("Outputs: ", outputs)
        outputs += perturbed_inputs
        # tf.print("Pert Outputs: ", outputs)
        # if self.use_bias:
        #     affine_bias = tf.random.normal(shape=tf.expand_dims(self.units, 0), stddev=bias_std)
        #     sign_output_bias = rademacher(tf.shape(affine_bias),
        #                                   dtype=inputs.dtype,
        #                                   seed=seed_stream())
        #     perturbed_bias = sign_output_bias * affine_bias
        #     bias += perturbed_bias

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)
        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable
        return outputs
