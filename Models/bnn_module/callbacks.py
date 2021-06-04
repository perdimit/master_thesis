import tensorflow as tf
import psutil

tfk = tf.keras

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================


class EpochDots(tfk.callbacks.Callback):
    """A simple callback that prints a "." every epoch, with occasional reports.
  Args:
    report_every: How many epochs between full reports
    dot_every: How many epochs between dots.
  """

    def __init__(self, report_every=100, dot_every=1):
        super(EpochDots, self).__init__()
        # self.report_every = report_every
        self.report_every = report_every
        self.dot_every = dot_every

    def on_epoch_end(self, epoch, logs=None):
        # filtered_logs = {key: logs[key] for key in ['loss', 'val_loss']}
        # Not ideal
        if 'val_kl_loss' in logs:
            logs.pop('kl_loss')
            logs.pop('kl_elbo')
            logs['kl_loss'] = logs.pop('val_kl_loss')
            logs['kl_elbo'] = logs.pop('val_kl_elbo')
            filtered_logs = {key: logs[key] for key in
                             ['elbo', 'val_elbo', 'kl_elbo', 'loss', 'val_loss', 'kl_loss', 'negloglik',
                              'val_negloglik']}
        else:
            filtered_logs = logs
        if epoch % self.report_every == 0:

            # mem = psutil.virtual_memory().used / 2 ** 30
            # tf.print("\n***  Memory usage: ~%sGb  ***" % mem)
            print()
            print('Epoch: {:d}, '.format(epoch), end='')
            for name, value in sorted(filtered_logs.items()):
                print('{}:{:0.4f}'.format(name, value), end=',  ')
            print()

        if epoch % self.dot_every == 0:
            print('.', end='')

    # def on_train_end(self, logs=None):
    #     # if epoch == self.epochs_tot-1:
    #         print("HEEEEEEEERE")
    #         if 'kl_loss' in logs:
    #             logs.pop('kl_loss')
    #             logs.pop('kl_elbo')
    #             logs['kl_loss'] = logs.pop('val_kl_loss')
    #             logs['kl_elbo'] = logs.pop('val_kl_elbo')

    # def on_train_end(self, logs=None):
    #     print(logs)
    #     if 'kl_loss' in logs:
    #         logs.pop('kl_loss')
    #         logs.pop('kl_elbo')
    #         logs['kl_loss'] = logs.pop('val_kl_loss')
    #         logs['kl_elbo'] = logs.pop('val_kl_elbo')


