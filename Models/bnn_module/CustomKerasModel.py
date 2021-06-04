import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tfk.layers
import numpy as np


class CustomKerasModel(tfk.Model):
    """
    In the standard keras model, predictions are made by one sample from one posterior predictive distribution(one forward pass), using model.predict(X).
    During training these are mainly used for reporting the metrics. So we here open for using the mean instead of a sample of the distribution.

    Additionally we want to be able to use the kl_weight as a hyperparameter. Where it may work as simulated annealing.

    Note that KL-loss is not dependent on the data, just the prior weights and pmf weights. However, the val_kl_loss is not the same as kl_loss, the former is the actual
    evlaution of the kl-divergence between prior and pmf, and the kl_loss is the weighted kl-divergence used for training. Both loss and val_loss, that is the ELBO losses,
    use kl_loss not val_kl_loss.
    """

    def __init__(self, kl_weight, use_mean, annealing_type='None', kappa=None, start_annealing=None, end_lin_annealing=None,
                 max_kl_weight=None, periods=None, batch_num=None, dtype='float64', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.use_mean = use_mean
        self.kl_metric = None
        self.negloglik_metric = None
        self.elbo_loss_metric = None
        self.batch_num = batch_num
        if None not in [start_annealing, max_kl_weight, end_lin_annealing] and annealing_type == 'linear':
            self.start_annealing = start_annealing
            self.end_lin_annealing = end_lin_annealing
            self.kl_weight_growth = tf.constant(
                (max_kl_weight - self.kl_weight) / (self.end_lin_annealing - self.start_annealing), dtype=dtype)
            self.max_kl_weight = max_kl_weight
            self.at = annealing_type
            self.kappa = None
        elif None not in [start_annealing, max_kl_weight, end_lin_annealing, periods] and annealing_type == 'sine':
            self.start_annealing = start_annealing
            self.end_lin_annealing = end_lin_annealing
            self.max_kl_weight = max_kl_weight
            self.periods = periods
            self.at = annealing_type
            self.kappa = None
        elif kappa is not None and annealing_type == 'exp':
            self.kappa = kappa
            self.batch_num = batch_num
            self.at = annealing_type
        else:
            self.start_annealing = None
            self.kappa = None
            self.at = 'None'
        if 'it' not in self.__dict__:
            self.it = tf.Variable(0.0, trainable=False, dtype=self.dtype)
        self.dtype_ = dtype
        if 'epoch' not in self.__dict__:
            self.epoch = tf.Variable(0.0, trainable=False, dtype=self.dtype)

    def special_compile(self, optimizer, loss, metrics, custom_metrics):
        # print(custom_metric)
        if custom_metrics is not None:
            for c in custom_metrics:
                if c == 'kl_loss':
                    self.kl_metric = tfk.metrics.Mean('kl_loss')
                elif c == 'kl_elbo':
                    self.kl_metric_elbo = tfk.metrics.Mean('kl_elbo')
                elif c == 'elbo':
                    self.elbo_loss_metric = tfk.metrics.Mean('elbo')
                elif c == 'negloglik':
                    self.negloglik_metric = tfk.metrics.Mean('negloglik')
                else:
                    pass
        # self.custom_metric_mean = tf.keras.metrics.Mean(custom_metric.name)
        super().compile(optimizer, loss, metrics)

    def compile(self, **kwargs):
        raise NotImplementedError("Please use special_compile()")

    def lin_growth_kl(self, it):
        it = it
        losses = []
        y = tf.multiply(tf.cast(it, dtype=self.dtype) - tf.constant(self.start_annealing, dtype=self.dtype),
                        self.kl_weight_growth) + tf.constant(self.kl_weight, dtype=self.dtype)
        for loss in self.losses:
            losses.append(loss * y)
        return losses

    def sinusoidal_kl(self, it):
        it = it
        b = (self.periods * np.pi) / (self.end_lin_annealing - self.start_annealing)
        w = self.kl_weight
        a_max = self.max_kl_weight - w
        y = a_max * tf.sin((it-self.start_annealing) * b) ** 2 + w
        losses = []
        # tf.print("self losses", tf.reduce_sum(self.losses))
        # tf.print("start kl weight", w)
        # tf.print("y", y)
        for loss in self.losses:
            losses.append(loss * y)
        return losses

    def exp_batch_decrease(self, it):
        prod = tf.multiply(tf.cast(it, dtype=self.dtype), tf.constant(self.kappa, dtype=self.dtype))
        expon = tf.exp(-prod, name='exp')
        losses = []
        for loss in self.losses:
            losses.append(loss * expon)
        return losses

    def return_loss(self, min_weight=True):
        # self.g_loss = self.losses
        losses = []
        for loss in self.losses:
            if min_weight:
                losses.append(loss * self.kl_weight)
            else:
                losses.append(loss * self.max_kl_weight)
        return losses

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_dist = self(x, training=True)
            if self.at == 'linear':
                kl_losses = tf.case([(tf.logical_and(self.it > self.start_annealing, self.it <= self.end_lin_annealing),
                                   lambda: self.lin_growth_kl(self.it)),
                                  (self.it > self.end_lin_annealing, lambda: self.return_loss(min_weight=False))],
                                 default=lambda: self.return_loss(min_weight=True), exclusive=False)
            elif self.at == 'sine':
                kl_losses = tf.case([(tf.logical_and(self.it > self.start_annealing, self.it <= self.end_lin_annealing),
                                   lambda: self.sinusoidal_kl(self.it)),
                                  (self.it > self.end_lin_annealing, lambda: self.return_loss(min_weight=True))],
                                 default=lambda: self.return_loss(min_weight=True), exclusive=False)
            elif self.at == 'exp':
                kl_losses = self.exp_batch_decrease(self.it)
            else:
                kl_losses = self.return_loss()
            loss = self.compiled_loss(y, y_dist, regularization_losses=kl_losses)

        if self.kl_metric is not None:
            self.kl_metric.update_state(tf.reduce_sum(kl_losses))
        if self.negloglik_metric is not None:
            negloglik = -y_dist.log_prob(tf.cast(y, tf.float64))
            self.negloglik_metric.update_state(negloglik)

        # Standard regularized metric
        if self.kl_metric_elbo is not None:
            self.kl_metric_elbo.update_state(tf.reduce_sum(self.losses) * (1 / self.batch_num))
        # Standard regularized metric
        if self.elbo_loss_metric is not None and self.batch_num is not None:
            elbo_loss = tf.reduce_sum(self.losses)*(1 / self.batch_num) - y_dist.log_prob(tf.cast(y, tf.float64))
            self.elbo_loss_metric.update_state(elbo_loss)

        y_dist = self(x, training=True)
        if self.use_mean:
            y_pred = y_dist.mean()
        else:
            y_pred = y_dist

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        self.it.assign_add(1)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        self.epoch.assign_add(1)
        if self.kappa is not None:
            self.it.assign(1)
        x, y = data
        # tf.print(self.current_kl_weight)
        y_dist = self(x, training=False)
        if self.use_mean:
            y_pred = y_dist.mean()
        else:
            y_pred = y_dist

        if self.kl_metric is not None:
            if self.at == 'linear':
                it = self.batch_num * self.epoch
                kl_losses = tf.case(
                    [(tf.logical_and(it > self.start_annealing, it <= self.end_lin_annealing), lambda: self.lin_growth_kl(it)),
                     (it > self.end_lin_annealing, lambda: self.return_loss(False))],
                    default=lambda: self.return_loss(True), exclusive=False)
            elif self.at == 'sine':
                it = self.batch_num * self.epoch
                kl_losses = tf.case(
                    [(tf.logical_and(it > self.start_annealing, it <= self.end_lin_annealing), lambda: self.sinusoidal_kl(it)),
                     (it > self.end_lin_annealing, lambda: self.return_loss(min_weight=True))],
                    default=lambda: self.return_loss(min_weight=True), exclusive=False)
            elif self.at == 'exp':
                kl_losses = self.exp_batch_decrease(tf.constant(self.batch_num, dtype=self.dtype))
            else:
                kl_losses = self.return_loss()
            self.kl_metric.update_state(tf.reduce_sum(kl_losses))
        if self.negloglik_metric is not None:
            negloglik = -y_dist.log_prob(tf.cast(y, tf.float64))
            self.negloglik_metric.update_state(negloglik)

        # Standard regularized metric
        if self.kl_metric_elbo is not None:
            self.kl_metric_elbo.update_state(tf.reduce_sum(self.losses) * (1 / self.batch_num))
        # Standard regularized metric
        if self.elbo_loss_metric is not None and self.batch_num is not None:
            elbo_loss = np.sum(self.losses) * (1 / self.batch_num) - y_dist.log_prob(tf.cast(y, tf.float64))
            self.elbo_loss_metric.update_state(elbo_loss)

        self.compiled_loss(y, y_dist, regularization_losses=kl_losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        metrics = super().metrics
        if self.kl_metric is not None:
            metrics.append(self.kl_metric)
        if self.negloglik_metric is not None:
            metrics.append(self.negloglik_metric)
        return metrics
