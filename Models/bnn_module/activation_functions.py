# from keras import backend as K
import tensorflow as tf
from tensorflow.keras.activations import elu

tfk = tf.keras
K = tfk.backend


def swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

def mish(x):
    soft = K.softplus(x)
    return x * K.tanh(soft)

def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)



def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
