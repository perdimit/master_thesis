import tensorflow as tf
import tensorflow_probability as tfp
import inspect

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions

def optimizer(optimizer_type_str, learning_rate):
    optim_bool = True
    while optim_bool:
        try:
            optimizer_type = getattr(tf.optimizers, optimizer_type_str)
            optimizer = optimizer_type(learning_rate=learning_rate)
            optim_bool = False
        except AttributeError:
            optim_bool = True
            print('The optimizer type chosen is not available.')
            print('Pick one from the following list\n')
            for name, obj in inspect.getmembers(tf.optimizers):
                if inspect.isclass(obj):
                    print(name)
            optimizer_type_str = input('\nEnter optimizer name: ')
            print('')
    return optimizer
