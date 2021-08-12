import itertools
import pandas as pd
import numpy as np
import psutil
import os
import pickle
import time
import datetime
import math as m
import sys
import shutil
import tensorflow as tf
import tensorflow_probability as tfp

print("TensorFlow version: ", tf.__version__)
print("TensorFlow-Probability version: ", tfp.__version__)

tfk = tf.keras
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tfk.layers

from . import data_handling as dh
from . import dict_handling as dcth
from . import error_handling as eh
from .BayesianNeuralNetwork import BayesianNeuralNetwork as BNN
from sklearn.metrics import r2_score as r2
from . import helper_files as hf
from .statistical_measures import standardized_residuals
from .statistical_measures import area_deviation_from_ideal_coverage
from .activation_functions import rmse


class Initialization:
    def __init__(self, data, target_name, data_test=None, prior=None, pmf=None,
                 test_fracs=None, affine_scaler='MinMaxScaler', log_scaling=True, affine_target_scaling=True,
                 model_name=None,
                 monitored_metrics=None, metrics_report='final', optim_by_unregularized_loss=True, mean_bool=True,
                 seed=42,
                 dtype="float64", strategy=None, manual_job_array=None, sort_permutations_by=None, latex_print=False,
                 mix_fix=True):
        """
        Prepares the data for use, scales, then trains and predicts. Class is specifically written for SLHA file type of data, but may work with other files as well.
        Only works with a single target. The class is intermingled with functions that stores all metrics for various initializations.
        Class is dependent on the classes BayesianNeuralNetwork, DistributionInitialization
        :param data: pd.DataFrame.
        Data set variable, must have both target and feature columns.
        :param target_name: string.
        Specificy which column name is the target.
        :param data_test: pd.DataFrame.
        Optional data set variable, must have both target and feature columns. Used for test runs after hyperparameter tuning has been performed.
        :param prior: Obj.
        Prior object from DistributionInitialization. Does not need to be passed in for hyperparameter tuning, but when loading parameters or a model it must be passed in.
        :param pmf: Obj.
        Posterior mean field object from DistributionInitialization. Does not need to be passed in for hyperparameter tuning, but when loading parameters or a model it must be passed in.
        :param test_fracs: List.
        Fractions of validation, and early stopping validation, in that order.
        :param affine_scaler: string.
        What type of affine scaler to use. Two types available. MinMaxScaler and StandardScaler.
        :param log_scaling: boolean.
        Whether or not to scale log-scale targets. Uses the natural logarithm.
        :param affine_target_scaling: boolean.
        Whether or not to scale the targets with the affine scaler (MinMax or StandardScaler.
        :param model_name: string.
        Optional, name of the model. This parameter is added to various strings when stored.
        :param monitored_metrics: List.
        A list of metrics that the user wants to monitor, that is, receive at the end of the run.
        :param metrics_report: string.
        What type of repoting. There are two, 'final' and 'training', where 'final' will only give metrics from validation, predicting with many samples.
        'training' gives both 'final' metrics and metrics from the training run. Such as the scaled losses.
        :param optim_by_unregularized_loss: boolean.
        'Particularly relevant for annealing. If true early stopping will monitor the ELBO that is not affected by annealing, just 1/batches as KL-weight. Valid whether annealing is on or not.
        If false then early stopping will monitor the annealed ELBO loss with a varying KL-weight in the case of annealing.
        :param mean_bool: boolean.
        During training this decides whether likelihood means (mean_bool=True) or random samples from the likelihood (mean_bool=False) should be used for loss evaluation.
        If the likelihood prediction samples are of the means then there will be a smaller variation in prediction samples than if its sampled from the likelihoods.
        If True it will also apply at prediction time, and thus for metrics. Note that the two options may give different results for predictions (especially for low sample_size).
        When mean_bool=True prediction samples effectively do not account for aleoteric uncertainty, which should be included for CAOS and CAUS calculation.
        However, with mean_bool=True less samples are needed for accurate predictions,
        and in addition aleoteric uncertainty for standardized residuals is calculated from the likelihood standard deviation, so this metric is unaffected.
        :param seed: int.
        Random number to initialize pseudorandom number generator.
        :param dtype: string.
        The data type. Must be a float.
        :param strategy: boolean.
        Whether MirroredStartegy is being used. Used to run more than one GPU, at most all GPUs on one HPC node.
        :param manual_job_array: List.
        If HPC job array is set to e.g. 2 and there are 4 permutations (models to try) we can pass in [3, 1], so that three will be trained
        by job 1, and one by job 2.
        :param sort_permutations_by:
        Sort permutations by hyperparameters. E.g. 'layers'. This is relevant to manual_job_array, say you want to train 2 shallow layers and 2 deep,
        than you would want 2 jobs on the deep models and possibly 1 for the shallow. If you sort the permutations then the 2 shallow layers will be first and then the deep
        making it easier to know how to set manual_job_array.
        :param latex_print:
        Whether to print out final metrics as dataframe or latex table (for copying) to screen.
        :param mix_fix:
        If true it produces a dataframe that says whether a cross section is bino, wino, higgsino dominated, convienient for plots. ONLY works for neutralino-neutralino.
        """
        self.seed = seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.df = data
        self.df_test = data_test
        if self.df_test is not None:
            self.testing = True
        else:
            self.testing = False
        if not isinstance(target_name, list):
            self.target_name = [target_name]
        else:
            self.target_name = target_name

        if self.target_name == "1000022_1000022_13000_NLO_1":
            self.mix_fix = mix_fix
        else:
            if mix_fix is True:
                print("\n *** mix_fix only works with 22-22, neutralino-neutralino. *** \n")
            self.mix_fix = False

        if None not in {prior, pmf}:
            self.prior = prior
            self.pmf = pmf
        self.save_name = model_name
        if test_fracs is None and not self.testing:
            self.data_fracs = [0.7, 0.1, 0.1]
        elif not self.testing:
            test_fracs.insert(0, 1 - sum(test_fracs))
            self.data_fracs = test_fracs
        elif self.testing:
            if test_fracs is not None and len(test_fracs) == 1:
                test_fracs.insert(0, 1 - test_fracs[0])
                self.data_fracs = test_fracs
            elif test_fracs is None:
                self.data_fracs = [0.8125, 0.1875]
            else:
                print("test fracs must have length 1")
                self.data_fracs = [0.8125, 0.1875]
        else:
            print("Did not recognize test_frac inputs or related")
        if affine_scaler in ['MinMaxScaler', 'StandardScaler', 'RobustScaler']:
            self.affine_scaler = affine_scaler
        else:
            print("Scaler is not available")
            self.affine_scaler = 'MinMaxScaler'
        self.log_scaling = log_scaling
        self.affine_target_scaling = affine_target_scaling
        self.dtype = dtype
        self.int_dtype = 'int32'
        self.strategy = strategy

        """***Global parameters used in various functions and the execution of functions. Explanation in relevant functions***"""
        self.output_scaler_obj = None
        self.input_scaler_obj = None
        self.X_train, self.X_vals, self.y_train, self.y_vals, self.X_es_val, self.y_es_val, self.X_val, self.y_val = None, None, None, None, None, None, None, None
        self.mix_df_train, self.mix_df_test, self.mix_df_val = None, None, None
        self.pred, self.pred_samples, self.pred_train, self.pred_train_samples = None, None, None, None
        self.val, self.mix_val, self.mix_train = None, None, None
        self.sigma_aleo, self.sigma_epis, self.sigma_mix = None, None, None
        self.eti, self.hdi = None, None
        self.store_scaled_errors = False
        self.y_sigma_aleo_scaled, self.y_sigma_epis_scaled, self.y_sigma_mix_scaled = None, None, None
        self.y_sigma_aleo_scaled_train, self.y_sigma_epis_scaled_train, self.y_sigma_mix_scaled_train = None, None, None

        # Prepare data, scaling etc.
        self.__prepare_data(self.testing)
        # Making validation pipeline
        self.__make_validation_pipeline(self.testing)

        self.scaler = None
        self.used_geometric_bool = False
        self.used_inverse_affine_bool = False

        self.mean_bool = mean_bool
        self.y_pred_samples = None
        self.y_pred_samples_stored = None
        self.y_pred_var_samples = None
        self.y_pred_es_samples = None
        self.y_pred_es_var_samples = None
        self.bnn = None
        self.model = None
        self.custom_model_bool = True

        self.perm_iteration = 0
        self.write_to_csv = False

        # Whether early_stopping monitors elbo-metric or elbo-loss
        if optim_by_unregularized_loss:
            optim = 'elbo'
        else:
            optim = 'loss'
        # What variable the table of model-metrics is sorted after."
        if self.testing:
            self.optim = 'test_' + optim
        else:
            self.optim = 'model_val_' + optim
        self.optim_bnn = 'val_' + optim
        self.optim_kl = 'kl_' + optim

        self.monitored_metrics = monitored_metrics
        if self.testing:
            self.metrics_report = 'testing'
            self.testing_metrics_dict, self.testing_metrics_keys = dcth.metrics_filter(self.monitored_metrics,
                                                                                       self.metrics_report,
                                                                                       optim=self.optim)
            self.training_metrics_keys, self.final_metrics_keys = None, None
        else:
            if metrics_report != 'testing':
                self.metrics_report = metrics_report
            else:
                print("Can't use 'testing' metrics without test data")
                self.metrics_report = 'training'
            self.training_metrics_dict, self.final_metrics_dict, self.training_metrics_keys, self.final_metrics_keys = \
                dcth.metrics_filter(self.monitored_metrics, self.metrics_report, optim=self.optim)
        self.results_df = None

        self.multiple_bool = False
        self.batch_sizes_mutable = None
        self.tf_train_data = None

        self.history_save = None
        self.current_best_model = {self.model, None, None}

        self.memory_usage_ls = None
        self.time_usage_ls = []
        self.save_best_model = False

        # HPC job array variables
        self.manual_job_array = manual_job_array
        if 'ARRAY_LENGTH' in os.environ and not self.testing:
            self.job_array_bool = True
            self.array_len = int(str(os.environ['ARRAY_LENGTH']).split('-')[-1])
            self.idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
            self.start = None
            self.end = None
        else:
            self.job_array_bool = False
        self.sort_permutations_by = sort_permutations_by
        self.latex_print = latex_print
        self.loaded_parameters = True

        # Run times
        self.nn_time = None
        self.bnn_time = None
        self.pred_time = None

    def __prepare_data(self, testing=False):
        """
        For hyperparameter tuning (testing=False) it splits the df into train, early stopping validation and model validation sets, and scales data accordingly.
        For testing (testing=True) it splits df into train and early stopping, takes in df_test separately, and scales all accordingly.
        If self.mix_fix = True a mixing dataframe is created that says whether xsec is bino, wino or higgsino dominated (only for neutralino-neutralino).

        :param testing: boolean.
        testing=False is hyperparameter tuning (main_training.py), and testing=True is testing (main_testing.py).
        """
        if self.mix_fix:
            mix_df = dh.get_mixing_df(self.df)
        else:
            mix_df = None

        # main_training.py use
        if not testing:
            X_ls, y_ls, mix_ls = dh.data_split(self.df,
                                               self.target_name,
                                               mix_df,
                                               test_fraction=[self.data_fracs[0], sum(self.data_fracs[1:])],
                                               seed=self.seed)

            if self.mix_fix:
                self.mix_df_train = mix_ls[0]
            else:
                mix_ls = [None, None]
            self.X_train, self.X_vals, self.y_train, self.y_vals = X_ls[0], X_ls[1], y_ls[0], y_ls[1]

            if self.log_scaling:
                self.y_train, self.y_vals = self.__log_scaling_target(self.y_train, self.y_vals)
            if self.affine_scaler is not None:
                self.X_train, self.X_vals, self.y_train, self.y_vals = self.__affine_scaling_data(self.X_train,
                                                                                                  self.X_vals,
                                                                                                  self.y_train,
                                                                                                  self.y_vals,
                                                                                                  scaler=self.affine_scaler,
                                                                                                  target_scaling=True)

            X_vals_ls, y_vals_ls, mix_vals_ls = dh.data_split(pd.concat((self.X_vals, self.y_vals), axis=1),
                                                              self.target_name,
                                                              mix_ls[1],
                                                              test_fraction=list(
                                                                  self.data_fracs[1:] / np.sum(self.data_fracs[1:])),
                                                              seed=self.seed)

            self.X_es_val, self.X_val = X_vals_ls[0], X_vals_ls[1]
            self.y_es_val, self.y_val = y_vals_ls[0], y_vals_ls[1]
            if self.mix_fix:
                self.mix_df_es_val, self.mix_df_val = mix_vals_ls[0], mix_vals_ls[1]
                self.mix_val = self.mix_df_val
                self.mix_train = self.mix_df_train
            print("Tuning train data fraction of total train data: %s%%" % round(100 * len(self.X_train) / len(self.df),
                                                                                 4))
            print("Early stopping validation data fraction of total train data: %s%%" % round(
                100 * len(self.X_es_val) / len(self.df), 4))
            print(
                "Validation data fraction of total train data:  %s%%" % round(100 * len(self.X_val) / len(self.df), 4))
            print("")
        # main_testing.py use
        else:
            if self.mix_fix:
                mix_df = dh.get_mixing_df(self.df)
                self.mix_df_test = dh.get_mixing_df(self.df_test)
                self.mix_df_train = dh.get_mixing_df(self.df)
                self.mix_val, self.mix_train = self.mix_df_test, self.mix_df_train
            df_test = self.df_test.sort_values(by=self.target_name)
            df = self.df.sort_values(by=self.target_name)
            self.X_test, self.y_test = df_test.drop(
                columns=self.target_name), df_test[self.target_name]
            self.X_train, self.y_train = df.drop(
                columns=self.target_name), df[self.target_name]
            if self.log_scaling:
                self.y_train, self.y_test = self.__log_scaling_target(self.y_train, self.y_test)
            if self.affine_scaler is not None:
                self.X_train, self.X_test, self.y_train, self.y_test = self.__affine_scaling_data(self.X_train,
                                                                                                  self.X_test,
                                                                                                  self.y_train,
                                                                                                  self.y_test,
                                                                                                  scaler=self.affine_scaler,
                                                                                                  target_scaling=True)

            X_ls, y_ls, _ = dh.data_split(pd.concat((self.X_train, self.y_train), axis=1),
                                          self.target_name,
                                          mix_df,
                                          test_fraction=[self.data_fracs[0], sum(self.data_fracs[1:])],
                                          seed=self.seed)

            self.X_train, self.X_es_val, self.y_train, self.y_es_val = X_ls[0], X_ls[1], y_ls[0], y_ls[
                1]

    def __make_validation_pipeline(self, testing=False):
        """
        Makes tensorflow related varilables to pass in to Keras for graph execution. It both makes a tensor tf.constant,
        and tf.data.datasets, where the latter is particularly necessary to speed up computation.

        :param testing: boolean.
        Whether validation/hyperparameter tuning run (False), or test run (train and test set).
        """

        self.tf_X_train = tf.constant(self.X_train, dtype=self.dtype)
        self.tf_y_train = tf.constant(self.y_train, dtype=self.dtype)

        if not testing:

            self.tf_X_val = tf.constant(self.X_val, dtype=self.dtype)
            self.tf_y_val = tf.constant(self.y_val, dtype=self.dtype)
            self.val_data = (self.tf_X_val, self.tf_y_val)
            self.tf_X_es_val = tf.constant(self.X_es_val, dtype=self.dtype)
            self.tf_y_es_val = tf.constant(self.y_es_val, dtype=self.dtype)
            self.es_val_data = (self.tf_X_es_val, self.tf_y_es_val)

            self.tf_data_val = tf.data.Dataset.from_tensor_slices((self.X_val.values, self.y_val.values))
            self.tf_data_val = self.tf_data_val.batch(len(self.X_val)).shuffle(len(self.X_val), seed=self.seed,
                                                                               reshuffle_each_iteration=False)
            self.tf_data_es_val = tf.data.Dataset.from_tensor_slices((self.X_es_val.values, self.y_es_val.values))
            self.tf_data_es_val = self.tf_data_es_val.batch(len(self.X_es_val)).shuffle(len(self.X_es_val),
                                                                                        seed=self.seed,
                                                                                        reshuffle_each_iteration=False)
        else:
            self.tf_X_test = tf.constant(self.X_test, dtype=self.dtype)
            self.tf_y_test = tf.constant(self.y_test, dtype=self.dtype)
            self.test_data = (self.tf_X_test, self.tf_y_test)

            self.tf_X_es_val = tf.constant(self.X_es_val, dtype=self.dtype)
            self.tf_y_es_val = tf.constant(self.y_es_val, dtype=self.dtype)
            self.es_val_data = (self.tf_X_es_val, self.tf_y_es_val)

            self.tf_data_es_val = tf.data.Dataset.from_tensor_slices((self.X_es_val.values, self.y_es_val.values))
            self.tf_data_es_val = self.tf_data_es_val.batch(len(self.X_es_val)).shuffle(len(self.X_es_val),
                                                                                        seed=self.seed,
                                                                                        reshuffle_each_iteration=False)

    def __make_batch_pipeline(self, batch_size):
        """
        As above, converts training set to tf.data.datasets pipeline for speed up. Also partitions data set into batches.

        :param batch_size: int.
         Sets the batch size for each mini-batch.
        """

        self.tf_train_data = tf.data.Dataset.from_tensor_slices((self.X_train.values, self.y_train.values))
        self.tf_train_data = self.tf_train_data.batch(batch_size).shuffle(len(self.X_train), seed=self.seed,
                                                                          reshuffle_each_iteration=True)
        print("Number of training batches: ", tf.data.experimental.cardinality(self.tf_train_data).numpy())

    def __log_scaling_target(self, y_train, y_val):
        """
        Does the very simple job of applying np.log() to the target pd.DataFrames in an overly convoluted manner.

        :param y_train: pd.DataFrame
        :param y_val: pd.DataFrame
        """

        if self.output_scaler_obj is not None:
            print("First run target_scaling_log then feature_scaling(scale_target=True)")
            return
        if not isinstance(self.target_name, list):
            self.target_name = [self.target_name]
        y_train = dh.logy_data(y_train, col_start=0, col_stop=len(self.target_name))
        y_val = dh.logy_data(y_val, col_start=0, col_stop=len(self.target_name))
        return y_train, y_val

    def __affine_scaling_data(self, X_train, X_vals, y_train, y_vals, scaler='StandardScaler', target_scaling=True):
        """
        Scales the data by the use of a sklearn scaler.

        :param X_train: pd.DataFrame.
        The feature training data.
        :param X_vals: pd.DataFrame.
        Either feature validation or test set.
        :param y_train:
        The target training data.
        :param y_vals:
        Either target validaiton or test set.
        :param scaler: string.
        The type of sklearn scaler. StandardScaler and MinMaxScaler are available.
        :param target_scaling: boolean.
        Whether to scale target values.
        """

        X_train, X_vals, y_train, y_vals, self.output_scaler_obj, self.input_scaler_obj = dh.scaling(
            X_train, X_vals, y_train, y_vals, scaling=scaler, target_scaling=target_scaling)
        return X_train, X_vals, y_train, y_vals

    def __multiple_runs_dict_prepare(self, param_dict):
        """
        Takes in parameter dict of lists of all hyperparameter configurations to run over and partitions it into a dict of dicts where each
        hyperparameter configuration is one dict. The length of the constructed permutations_dicts is the number of models in total.

        :param param_dict: dict of lists.
        Each list in the dict should contain all parameters that one wants to be try. e.g. {'kl_weight': [0.0001, 0.001], 'learning_rate: [0.1, 0.01, 0.001]'}
        """

        batch_sizes = param_dict.get("batch_size")
        if batch_sizes is None or batch_sizes[0] is None:
            param_dict["batch_size"] = [len(self.X_train)]
            self.batch_sizes_mutable = [len(self.X_train)]
        else:
            self.batch_sizes_mutable = batch_sizes

        keys, values = zip(*param_dict.items())
        self.permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if self.sort_permutations_by is None:
            if len(param_dict["batch_size"]) > 1:
                self.permutations_dicts = dh.insertion_sort(self.permutations_dicts)
        else:
            self.permutations_dicts = dh.insertion_sort(self.permutations_dicts, sort_by=self.sort_permutations_by)

        if self.job_array_bool:
            self.__job_array_check()

    def multiple_runs(self, param_dict, write_to_csv=True, overwrite=False, save_best_model=True,
                      save_all_model_hyperparams=True,
                      monitor_values='final'):
        """
        The main hyperparameter tuning function. This function loops over all the permutations of hyperparameters and distributions and trains a model for each.

        :param param_dict: dict of lists.
        Each list in the dict contains all the parameters for that given paramater that should be tried. e.g. 'learning_rate: [0.1, 0.01, 0.001]'
        :param write_to_csv: boolean.
        Decides if to store results in a csv file.
        :param overwrite: boolean.
        Whether to overwrite previous csv, saved model or parameters files. If set to false a new one is created with _number added.
        :param save_best_model: boolean.
        Whether to save all models or the best model.
        :param save_all_model_hyperparams:
        Whether to save all hyperparameters or not. Relevant for loading some configuration for training when doing main_testing.py.
        :param monitor_values:
        See class doc string (on top) on 'final' vs 'training'.
        """

        self.multiple_bool = True
        self.write_to_csv = write_to_csv
        self.overwrite = overwrite
        self.monitor_str = monitor_values
        self.save_best_model = save_best_model
        self.save_all_model_hyperparams = save_all_model_hyperparams

        if "distributions" in param_dict:
            param_dict = dcth.distribute_distributions_parameters(param_dict)

        self.__multiple_runs_dict_prepare(param_dict)

        self.memory_usage_ls = np.zeros(len(self.permutations_dicts))
        best_val = 0
        for i, p_dict in enumerate(self.permutations_dicts):
            t = time.time()
            self.perm_iteration = i
            param_dict = {key: p_dict[key] for key in p_dict if key not in ['distributions']}
            dist_dict = dcth.printable_distributions_dict(p_dict, p_dict['distributions'][0]['prior'].dist,
                                                          p_dict['distributions'][1]['pmf'].dist)
            printable_dict = {**param_dict, **dist_dict}

            print("\n****************")
            print("Permutation %s/%s - Parameters %s" % (i + 1, len(self.permutations_dicts), printable_dict))
            print("****************\n")

            batch_size = p_dict.get("batch_size")
            if batch_size in self.batch_sizes_mutable:
                self.__make_batch_pipeline(batch_size)
                self.batch_sizes_mutable.remove(batch_size)
            else:
                self.__make_batch_pipeline(batch_size)
            self.run(**p_dict)

            df_print = pd.DataFrame(
                [(key, elem[self.perm_iteration]) for (key, elem) in self.training_metrics_dict.items()],
                columns=['metrics', 'values'])
            print("\nBest results this run: \n", df_print)

            model = self.model

            if self.save_all_model_hyperparams:
                self.save_model_params_only()

            # Stores current best parameters for saving.
            if i == 0:
                best_val = self.training_metrics_dict[self.optim][0]
                if save_best_model:
                    bnn_best_epoch = self.bnn_best_epoch
                    if hasattr(self.prior, 'mle'):
                        mle = self.prior.mle.copy()
                        nn_best_epoch = self.prior.best_epoch
                    elif hasattr(self.pmf, 'mle'):
                        mle = self.prior.mle.copy()
                        nn_best_epoch = self.prior.best_epoch
                    else:
                        mle = None
                        nn_best_epoch = None
                    self.current_best_model = [model, self.permutations_dicts[i], self.history_save.copy(), mle,
                                               nn_best_epoch, bnn_best_epoch]
            else:
                new_val = self.training_metrics_dict[self.optim][-1]
                if new_val < best_val:
                    print("New best %s: %s" % (self.optim, new_val))
                    print("Old best %s: %s" % (self.optim, best_val))
                    print("")
                    best_val = new_val
                    if save_best_model:
                        bnn_best_epoch = self.bnn_best_epoch
                        if hasattr(self.prior, 'mle'):
                            mle = self.prior.mle.copy()
                            nn_best_epoch = self.prior.best_epoch
                        elif hasattr(self.pmf, 'mle'):
                            mle = self.prior.mle.copy()
                            nn_best_epoch = self.prior.best_epoch
                        else:
                            mle = None
                        self.current_best_model = [model, self.permutations_dicts[i], self.history_save.copy(), mle,
                                                   nn_best_epoch, bnn_best_epoch]
                else:
                    print("Current best %s: %s" % (self.optim, best_val))

            del self.model
            K.clear_session()
            tf.compat.v1.reset_default_graph()

            mem = psutil.virtual_memory().used / 2 ** 30
            print("\n***  Memory usage: ~%sGb  ***" % mem)
            self.memory_usage_ls[i] = mem
            run_time = time.time() - t
            self.time_usage_ls.append(str(datetime.timedelta(seconds=run_time)))

        self.__save_and_print(save_best_model)
        self.multiple_bool = False
        return pd.DataFrame(self.permutations_dicts), self.results_df

    def __job_array_check(self):
        """
        Method relating to job_array on HPC. Tells each job (a number of CPUs/GPUs) which and how many models to run through of the configurations in self.permutation_dicts.
        """

        if self.array_len > len(self.permutations_dicts):
            print(
                "Set job array length to less(or equal) than the total number of model permutations %s, and try again." % len(
                    self.permutations_dicts))
            sys.exit()
        elif self.manual_job_array is not None:
            if sum(self.manual_job_array) != len(self.permutations_dicts):
                print("Please set the number of jobs in manual_job_array equal to the number of model permutations")
            models_per_job = self.manual_job_array[self.idx]
            self.start = self.idx * models_per_job
            self.end = (self.idx + 1) * models_per_job
            self.permutations_dicts = self.permutations_dicts[self.start:self.end]
        else:
            models_per_job = m.ceil(len(self.permutations_dicts) / self.array_len)
            self.start = self.idx * models_per_job
            self.end = (self.idx + 1) * models_per_job
            if self.start + models_per_job > len(self.permutations_dicts):
                self.end = len(self.permutations_dicts)
            self.permutations_dicts = self.permutations_dicts[self.start:self.end]

    def test_run_with_params(self, parameters_directory, sample_size=None, save_model=True, overwrite=True,
                             write_to_csv=True):
        """
        Method used for running main_testing.py with df and df_test. This is for after hyperparameter tuning.
        Allows one to pass in saved parameters from validation model.

        :param parameters_directory: string.
        The folder where the parameters lie.
        :param sample_size: int.
        The sample size to use for predictions.
        :param save_model: boolean.
        Whether to save model or not.
        :param overwrite: boolean.
        Whether to overwrite existing files they already exists.
        :param write_to_csv: boolean.
        Whether to create a csv file with the results.
        """

        self.overwrite = overwrite
        self.write_to_csv = write_to_csv

        if sample_size is None:
            self.load_parameters(dirname_bnn=parameters_directory)
        else:
            self.load_parameters(dirname_bnn=parameters_directory, sample_size=sample_size)
        t = time.time()
        p_dict = self.permutations_dicts[0]
        batch_size = p_dict.get("batch_size")
        self.__make_batch_pipeline(batch_size)
        self.run(**p_dict)

        run_time = time.time() - t
        self.time_usage_ls.append(str(datetime.timedelta(seconds=run_time)))

        self.__save_and_print(save_model=save_model)
        self.multiple_bool = False

        del self.model
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        return pd.DataFrame(self.permutations_dicts), self.results_df

    def run(self, layers, epochs=1000, patience=None, batch_size=32, optimizer_type='Adam', learning_rate=0.01,
            activation='sigmoid',
            kl_weight=None, use_flipout=False, soft_coef_ll=1, alpha=0.01, annealing_type='None', kappa=None,
            start_annealing=None, max_kl_weight=None, annealing_phase=None, periods=None,
            sample_size=200, loaded_model=False, **kwargs):
        """
        Builds and runs a BNN given a set of hyperparameters. See main_training.py for most of the parameter explanations.

        :param soft_coef_ll:
        A coefficient to modulate the standard deviation of the likelihood.
        :param alpha: float.
        ELU and LeakyRelu has an alpha parameter which can be set by passing in alpha. Add it to the parameter dictionary passed into multiple_runs.
        """

        self.epochs = epochs
        self.patience = patience
        self.use_flipout = use_flipout

        # deterministic/empirical neural network parameters
        nn_inputs = {'train_data': self.tf_train_data, 'layers': layers, 'epochs': epochs, 'patience': patience,
                     'batch_size': batch_size, 'optimizer_type': optimizer_type, 'learning_rate': learning_rate,
                     'activation': activation, 'val_data': self.tf_data_es_val}


        if kwargs['distributions'] is not None:
            prior_params, pmf_params, prior_arg_keys, pmf_arg_keys, set_prior, set_pmf, self.prior, self.pmf = dcth.get_distributions_parameters(
                kwargs, nn_inputs)
        if not loaded_model:
            # Setting prior and posterior and obtaining their respective parameters
            set_prior(**prior_params)
            set_pmf(**pmf_params)

        optimizer = eh.optimizer(optimizer_type, learning_rate)
        batch_num = m.ceil(self.X_train.shape[0] / batch_size)
        if kl_weight is None:
            kl_weight = tf.constant(1 / batch_num, dtype=self.dtype)
        else:
            kl_weight = tf.constant(kl_weight, dtype=self.dtype)

        # annealing settings
        if annealing_type == 'exp' and None not in [kappa]:
            start_annealing = None
            end_lin_annealing = None
        elif annealing_type in ['linear', 'sine'] and None not in [start_annealing, annealing_phase, max_kl_weight,
                                                                   kl_weight]:
            end_lin_annealing = (start_annealing + annealing_phase) * batch_num
            start_annealing = start_annealing * batch_num
        else:
            kappa = None
            start_annealing = None
            end_lin_annealing = None
            if annealing_type in ['exp', 'linear', 'sine']:
                print(
                    "Some of the parameters for %s was not correctly set. No annealing will be used." % annealing_type)
            annealing_type = 'None'

        # defining the BNN model using BayesianNeuralNetwork.py class
        self.bnn = BNN(layers, self.pmf, self.prior, kl_loss_weight=kl_weight, activation=activation,
                       use_flipout=self.use_flipout, soft_coef_ll=soft_coef_ll, alpha=alpha,
                       annealing_type=annealing_type,
                       kappa=kappa, start_annealing=start_annealing, end_lin_annealing=end_lin_annealing,
                       batch_num=batch_num, max_kl_weight=max_kl_weight, periods=periods, seed=self.seed,
                       dtype=self.dtype)

        # If custom_model=True in self.bnn.build_model CustomKerasModel.py is used.
        # This is necessary for annealing but also works without annealing, thus should always be set to true.
        if self.strategy is not None:
            with self.strategy.scope():
                self.bnn.build_model(custom_model=self.custom_model_bool, use_mean=True)
                self.bnn.compile(optimizer=optimizer, loss=self.__negloglik,
                                 metrics=[tfk.metrics.RootMeanSquaredError(name='rmse')],
                                 custom_metrics=['kl_loss', 'kl_elbo', 'negloglik', 'elbo'])
        else:
            self.bnn.build_model(custom_model=self.custom_model_bool, use_mean=True)
            self.bnn.compile(optimizer=optimizer, loss=self.__negloglik,
                             metrics=[tfk.metrics.RootMeanSquaredError(name='rmse')],
                             custom_metrics=['kl_loss', 'kl_elbo', 'negloglik', 'elbo'])
        self.model = self.bnn.model

        if hasattr(self.prior, 'time_used'):
            self.nn_time = self.prior.time_used
        elif hasattr(self.pmf, 'time_used'):
            self.nn_time = self.pmf.time_used
        else:
            self.nn_time = None

        if not loaded_model:
            t = time.time()
            history = self.bnn.fit(self.tf_train_data, self.tf_data_es_val, self.epochs, self.patience, verbose=0,
                                   monitor=self.optim_bnn)
            self.history_save = history
            run_time = time.time() - t
            self.bnn_time = str(datetime.timedelta(seconds=round(run_time)))
        else:
            history = self.__load_weights()

        if self.testing:
            X = self.tf_X_test
        else:
            X = self.tf_X_val
        t = time.time()
        # Creates predictions on data set model validation or test. This are used by the variousu __append functions below.
        self.y_pred, self.y_pred_samples, self.y_pred_var_samples = self.predict(X,
                                                                                 sample_size=tf.constant(sample_size,
                                                                                                         dtype=self.int_dtype),
                                                                                 mean=self.mean_bool)
        run_time = time.time() - t
        self.pred_time = str(datetime.timedelta(seconds=round(run_time)))

        if self.testing:
            self.bnn_best_epoch = np.argmin(history[self.optim_bnn]) + 1
            self.__append_testing_metrics(history, sample_size=sample_size)
        else:
            self.bnn_best_epoch = np.argmin(history[self.optim_bnn]) + 1
            self.__append_final_metrics(history, sample_size=sample_size)
            self.__append_training_metrics(history)


    def __append_training_metrics(self, history):
        """
        This function appends various metrics to the training_metrics_dict dictionary taken from Keras' training-history.
        Used during model selection.

        :param history: dict.
        """

        stopped_epoch = history['stopped_epoch']
        model_val_negloglik = self.__negloglik(self.y_val, self.model(self.X_val)).numpy()
        best_epoch = self.bnn_best_epoch - 1
        self.training_metrics_dict.setdefault('stopped_epoch', []).append(stopped_epoch)
        self.training_metrics_dict.setdefault('best_epoch', []).append(best_epoch)

        self.training_metrics_dict.setdefault('kl_loss', []).append(history['kl_loss'][best_epoch])
        self.training_metrics_dict.setdefault('model_val_loss', []).append(
            model_val_negloglik + history['kl_loss'][best_epoch])
        self.training_metrics_dict.setdefault('model_val_negloglik', []).append(model_val_negloglik)

        self.training_metrics_dict.setdefault('kl_elbo', []).append(history['kl_elbo'][best_epoch])
        self.training_metrics_dict.setdefault('model_val_elbo', []).append(
            model_val_negloglik + history['kl_elbo'][best_epoch])
        for i, m in enumerate(self.training_metrics_keys):
            m_ = m.split('_')
            try:
                if m_[0] == 'es':
                    # early stopping validation metrics
                    self.training_metrics_dict.setdefault(m, []).append(history[m_[1] + '_' + m_[2]][best_epoch])
                elif m_[0] == 'train':
                    # early stopping train metrics
                    self.training_metrics_dict.setdefault(m, []).append(history[m_[1]][best_epoch])
            except KeyError:
                print("%s is not available for training metrics" % m)
                pass

    #  Predicting and inverting
    def __append_final_metrics(self, history, sample_size=100):
        """
        This function appends various metrics to the final_metrics_dict dictionary.
        Unlike __append_training_metrics, this one includes metrics on the final inverted prediction results.
        It includes all uncertainty metrics and residual type metrics (R2, RMSE).
        Used during model selection.

        :param history: dict.
        Metric and loss history from the testing run.

        """

        if self.affine_scaler:
            y_pred_train, y_pred_train_samples, _ = self.predict(self.tf_X_train, sample_size=sample_size,
                                                                 mean=self.mean_bool)
            y_pred_es, _, _ = self.predict(self.tf_X_es_val, sample_size=sample_size, mean=self.mean_bool)
            X_ls = dh.inverse_affine_scaling(
                [self.X_train.copy(), self.X_es_val.copy(), self.X_val.copy()],
                scaler_obj=self.input_scaler_obj, is_input=True)
            X_train, X_es_val, X_val = X_ls[0], X_ls[1], X_ls[2]
            y_ls = dh.inverse_affine_scaling(
                [self.y_train.copy(), self.y_es_val.copy(), self.y_val.copy()],
                scaler_obj=self.output_scaler_obj, is_input=False)
            y_train, y_es_val, y_val = y_ls[0], y_ls[1], y_ls[2]
            y_ls_pred = dh.inverse_affine_scaling(
                [y_pred_train, self.y_pred,
                 y_pred_es, y_pred_train_samples, self.y_pred_samples],
                scaler_obj=self.output_scaler_obj, is_input=False)
            y_pred_train, y_pred, y_pred_es, y_pred_train_samples, y_pred_samples = y_ls_pred[0], y_ls_pred[1], \
                                                                                    y_ls_pred[2], y_ls_pred[3], \
                                                                                    y_ls_pred[4]

            y_pred_var_samples = dh.inverse_affine_scaling([self.y_pred_var_samples], scaler_obj=self.output_scaler_obj,
                                                           is_input=False, is_variance=True)
            y_sigma_aleo, y_sigma_epis, y_sigma_mix = self.bnn.predict_sigmas(y_pred_samples, y_pred_var_samples,
                                                                              inv=True,
                                                                              n=1)
            if self.store_scaled_errors:
                self.y_sigma_aleo_scaled, self.y_sigma_epis_scaled, self.y_sigma_mix_scaled = self.bnn.predict_sigmas(
                    y_pred_samples, y_pred_var_samples,
                    inv=False,
                    n=1)

        if self.log_scaling:
            tn = self.target_name[0]
            y_ls_0 = dh.inverse_log_scaling([y_train, y_es_val, y_val])
            y_train, y_es_val, y_val = y_ls_0[0][tn].values, y_ls_0[1][tn].values, y_ls_0[2][tn].values
            y_ls_1 = dh.inverse_log_scaling([y_pred_train, y_pred_es, y_pred])
            y_pred_train, y_pred_es, y_pred = y_ls_1[0], y_ls_1[1], y_ls_1[2]
            y_ls_2 = dh.inverse_log_scaling([y_pred_train_samples, y_pred_samples])
            y_pred_train_samples, y_pred_samples = y_ls_2[0], y_ls_2[1]

        # Storing predictions and uncertainties for get functions.
        self.pred, self.pred_samples, self.pred_train, self.pred_train_samples = y_pred, y_pred_samples, y_pred_train, y_pred_train_samples
        self.sigma_aleo, self.sigma_epis, self.sigma_mix = y_sigma_aleo, y_sigma_epis, y_sigma_mix
        self.val = pd.DataFrame(data=y_val, columns=[self.target_name], index=self.X_val.index)

        stopped_epoch = history['stopped_epoch']
        best_epoch = self.bnn_best_epoch - 1
        model_val_negloglik = self.__negloglik(self.y_val, self.model(self.X_val)).numpy()
        self.final_metrics_dict.setdefault(self.optim_kl, []).append(history[self.optim_kl][best_epoch])
        self.final_metrics_dict.setdefault(self.optim + '(scaled)', []).append(
            model_val_negloglik + history[self.optim_kl][best_epoch])

        self.final_metrics_dict.setdefault('stopped_epoch', []).append(stopped_epoch)
        self.final_metrics_dict.setdefault('best_epoch', []).append(best_epoch)

        vals = [[y_pred_train, y_train, X_train],
                [y_pred_es, y_es_val, X_es_val],
                [y_pred, y_val, X_val]]
        type_str = ['train', 'es_val', 'model_val']

        for i, ls in enumerate(vals):
            self.final_metrics_dict.setdefault(type_str[i] + '_negloglik', []).append(
                self.__negloglik(ls[1], self.model(ls[2])).numpy())
            self.final_metrics_dict.setdefault(type_str[i] + '_rmse', []).append(rmse(ls[0], ls[1]).numpy())
            self.final_metrics_dict.setdefault(type_str[i] + '_r2', []).append(r2(ls[0], ls[1]))

        # standardized residuals
        mix_residuals, mix_res_stddev, mix_res_mean = standardized_residuals(y_val, y_pred, y_sigma_mix)
        self.final_metrics_dict.setdefault('mix_res_stddev', []).append(mix_res_stddev)
        self.final_metrics_dict.setdefault('mix_res_mean', []).append(mix_res_mean)
        epis_residuals, epis_res_stddev, epis_res_mean = standardized_residuals(y_val, y_pred, y_sigma_epis)
        self.final_metrics_dict.setdefault('epis_res_stddev', []).append(epis_res_stddev)
        self.final_metrics_dict.setdefault('epis_res_mean', []).append(epis_res_mean)
        aleo_residuals, aleo_res_stddev, aleo_res_mean = standardized_residuals(y_val, y_pred, y_sigma_aleo)
        self.final_metrics_dict.setdefault('aleo_res_stddev', []).append(aleo_res_stddev)
        self.final_metrics_dict.setdefault('aleo_res_mean', []).append(aleo_res_mean)

        # CAOS and CAUS scores.
        coverage_deviation_score, area_over, area_under = area_deviation_from_ideal_coverage(y_pred_samples, y_val,
                                                                                             interval_type='ETI',
                                                                                             resolution=100)
        self.final_metrics_dict.setdefault('cds', []).append(coverage_deviation_score)
        self.final_metrics_dict.setdefault('caos', []).append(area_over)
        self.final_metrics_dict.setdefault('caus', []).append(area_under)

        if self.nn_time is not None:
            self.final_metrics_dict.setdefault('nn_time', []).append(self.nn_time)
        self.final_metrics_dict.setdefault('bnn_time', []).append(self.bnn_time)
        self.final_metrics_dict.setdefault('pred_time', []).append(self.pred_time)

    def __append_testing_metrics(self, history, sample_size=100):
        """
        Function similar to the __append_final_metrics, but deals with metrics corresponding to a test run.
        Inverts all predictions and data appends the metrics to a testing metrics dictionary.

        :param history: dict.
        Metric and loss history from the testing run.
        :param sample_size: int.
        Number of samples to take when predicting on the train set.
        """

        if self.affine_scaler:
            y_pred_train, y_pred_train_samples, y_pred_train_var_samples = self.predict(self.tf_X_train, sample_size=sample_size,
                                                                 mean=self.mean_bool)
            X_ls = dh.inverse_affine_scaling(
                [self.X_train.copy(), self.X_test.copy()],
                scaler_obj=self.input_scaler_obj, is_input=True)
            X_train, X_test = X_ls[0], X_ls[1]
            y_ls = dh.inverse_affine_scaling(
                [self.y_train.copy(), self.y_test.copy()],
                scaler_obj=self.output_scaler_obj, is_input=False)
            y_train, y_test = y_ls[0], y_ls[1]
            y_ls_pred = dh.inverse_affine_scaling(
                [y_pred_train, self.y_pred, y_pred_train_samples, self.y_pred_samples],
                scaler_obj=self.output_scaler_obj, is_input=False)
            y_pred_train, y_pred, y_pred_train_samples, y_pred_samples = y_ls_pred[0], y_ls_pred[1], \
                                                                         y_ls_pred[2], y_ls_pred[3]

            y_pred_var_samples = dh.inverse_affine_scaling([self.y_pred_var_samples], scaler_obj=self.output_scaler_obj,
                                                           is_input=False, is_variance=True)
            y_sigma_aleo, y_sigma_epis, y_sigma_mix = self.bnn.predict_sigmas(y_pred_samples.copy(), y_pred_var_samples.copy(),
                                                                              inv=True,
                                                                              n=1)
            if self.store_scaled_errors:
                self.y_sigma_aleo_scaled, self.y_sigma_epis_scaled, self.y_sigma_mix_scaled = self.bnn.predict_sigmas(
                    y_pred_samples, y_pred_var_samples,
                    inv=False,
                    n=1)
                y_pred_train_var_samples = dh.inverse_affine_scaling([y_pred_train_var_samples],
                                                               scaler_obj=self.output_scaler_obj,
                                                               is_input=False, is_variance=True)
                self.y_sigma_aleo_scaled_train, self.y_sigma_epis_scaled_train, self.y_sigma_mix_scaled_train = self.bnn.predict_sigmas(
                    y_pred_train_samples, y_pred_train_var_samples,
                    inv=False,
                    n=1)
        if self.log_scaling:
            tn = self.target_name[0]
            y_ls_0 = dh.inverse_log_scaling([y_train, y_test])
            y_train, y_test = y_ls_0[0][tn].values, y_ls_0[1][tn].values
            y_ls_1 = dh.inverse_log_scaling([y_pred_train, y_pred])
            y_pred_train, y_pred = y_ls_1[0], y_ls_1[1]
            y_ls_2 = dh.inverse_log_scaling([y_pred_train_samples, y_pred_samples])
            y_pred_train_samples, y_pred_samples = y_ls_2[0], y_ls_2[1]

        stopped_epoch = len(history['epoch']) - 1
        best_epoch = self.bnn_best_epoch - 1

        self.pred, self.pred_samples, self.pred_train, self.pred_train_samples = y_pred, y_pred_samples, y_pred_train, y_pred_train_samples
        self.sigma_aleo, self.sigma_epis, self.sigma_mix = y_sigma_aleo, y_sigma_epis, y_sigma_mix
        self.val = pd.DataFrame(data=y_test, columns=[self.target_name], index=self.X_test.index)

        # scaled
        test_negloglik = self.__negloglik(self.y_test, self.model(self.X_test)).numpy()
        train_negloglik = self.__negloglik(self.y_train, self.model(self.X_train)).numpy()
        self.testing_metrics_dict.setdefault('kl_elbo', []).append(history['kl_elbo'][best_epoch])
        self.testing_metrics_dict.setdefault('test_elbo' + '(scaled)', []).append(
            test_negloglik + history['kl_elbo'][best_epoch])
        self.testing_metrics_dict.setdefault('train_elbo' + '(scaled)', []).append(
            train_negloglik + history['kl_elbo'][best_epoch])

        self.testing_metrics_dict.setdefault('kl_loss', []).append(history['kl_loss'][best_epoch])
        self.testing_metrics_dict.setdefault('test_loss' + '(scaled)', []).append(
            test_negloglik + history['kl_loss'][best_epoch])
        self.testing_metrics_dict.setdefault('train_loss' + '(scaled)', []).append(history['loss'][best_epoch])

        # unscaled
        self.testing_metrics_dict.setdefault('stopped_epoch', []).append(stopped_epoch)
        self.testing_metrics_dict.setdefault('best_epoch', []).append(best_epoch)
        vals = [[y_pred_train, y_train, X_train],
                [y_pred, y_test, X_test]]
        type_str = ['train', 'test']
        for i, ls in enumerate(vals):
            self.testing_metrics_dict.setdefault(type_str[i] + '_negloglik', []).append(
                self.__negloglik(ls[1], self.model(ls[2])).numpy())
            self.testing_metrics_dict.setdefault(type_str[i] + '_rmse', []).append(rmse(ls[0], ls[1]).numpy())
            self.testing_metrics_dict.setdefault(type_str[i] + '_r2', []).append(r2(ls[0], ls[1]))

        # standardized residuals
        mix_residuals, mix_res_stddev, mix_res_mean = standardized_residuals(y_test, y_pred, y_sigma_mix)
        self.testing_metrics_dict.setdefault('mix_res_stddev', []).append(mix_res_stddev)
        self.testing_metrics_dict.setdefault('mix_res_mean', []).append(mix_res_mean)

        # CAOS and CAUS
        coverage_deviation_score, area_over, area_under = area_deviation_from_ideal_coverage(y_pred_samples, y_test,
                                                                                             interval_type='ETI',
                                                                                             resolution=100)
        self.testing_metrics_dict.setdefault('cds', []).append(coverage_deviation_score)
        self.testing_metrics_dict.setdefault('caos', []).append(area_over)
        self.testing_metrics_dict.setdefault('caus', []).append(area_under)

    def predict(self, X_test=None, sample_size=100, mean=False):
        """
        Main prediction function used in model selection.

        :param X_test: pd.Dataframe or tensor.
        :param sample_size: int.
        :param mean: boolean.
        Whether to get means from likelihoods or samples from likelihoods.
        """

        if self.bnn is not None:
            if X_test is None:
                X_test = self.X_val
            y_pred_samples, y_pred_var_samples = self.bnn.predict(X_test, sample_size, mean)
            if sample_size > 1:
                y_pred = tf.reduce_mean(y_pred_samples, axis=0)
            else:
                y_pred = y_pred_samples
            return y_pred.numpy(), y_pred_samples.numpy(), y_pred_var_samples.numpy()
        else:
            print("Please train or load a model first.")
            return

    def predict_singles(self, X_single=None, sample_dist_size=100, sample_size=100):
        """
        Function to predict a single data point/xsec. Let's you sample multiple times from each sampled likelihood distribution (not the case for other prediction functions).
        Have only been used for checking validity for the other functions, and not part of the greater framework.

        :param X_single: pd.Dataframe or tensor
        :param sample_dist_size: number of likelihoods (forward passes)
        :param sample_size: number of samples from each likelihood
        """

        if self.bnn is not None:
            if X_single is None:
                X_single = self.X_val.sample(n=1)
                return self.bnn.predict_singles(X_single, sample_dist_size, sample_size)
            else:
                return self.bnn.predict_singles(X_single, sample_dist_size, sample_size)

        else:
            print("Please train or load a model first.")
            return

    def predict_standalone(self, X_test, sample_size=1000, invert_log=False, invert_errors=False, mean=False):
        """
        Function used to predict separate from the greater framework, but with the possibility of inverting the predictions.

        :param X_test: pd.Dataframe or tensor.
        :param sample_size: int.
        :param invert_log: boolean.
        invert all target and prediction based values
        :param invert_errors: boolean.
        invert the uncertainty estimates or not
        :param mean: boolean.
        get means from likelihoods or samples from likelihoods.

        """


        if self.bnn is not None:
            t = time.time()
            y_pred_samples, y_pred_var_samples = self.bnn.predict(X_test, sample_size, mean)
            if sample_size > 1:
                y_pred = tf.reduce_mean(y_pred_samples, axis=0)
            else:
                y_pred = y_pred_samples
            y_pred, y_pred_samples, y_pred_var_samples = y_pred.numpy(), y_pred_samples.numpy(), y_pred_var_samples.numpy()

            y_ls_pred = dh.inverse_affine_scaling(
                [y_pred, y_pred_samples],
                scaler_obj=self.output_scaler_obj, is_input=False)
            y_pred, y_pred_samples = y_ls_pred[0], y_ls_pred[1]
            y_ls = dh.inverse_affine_scaling(
                [self.y_test.copy()],
                scaler_obj=self.output_scaler_obj, is_input=False)
            y_test = y_ls[0]

            y_pred_var_samples = dh.inverse_affine_scaling([y_pred_var_samples], scaler_obj=self.output_scaler_obj,
                                                           is_input=False, is_variance=True)
            y_sigma_aleo, y_sigma_epis, y_sigma_mix = self.bnn.predict_sigmas(y_pred_samples, y_pred_var_samples,
                                                                              inv=invert_errors,
                                                                              n=1)
            if invert_log:
                y_ls_0 = dh.inverse_log_scaling([y_test])
                y_test = y_ls_0[0]
                y_ls_1 = dh.inverse_log_scaling([y_pred])
                y_pred = y_ls_1[0]
                y_ls_2 = dh.inverse_log_scaling([y_pred_samples])
                y_pred_samples = y_ls_2[0]

            pred_time = time.time() - t
            return y_pred, y_pred_samples, y_test, y_sigma_mix, y_sigma_epis, y_sigma_aleo, pred_time
        else:
            print("Please train or load a model first.")
            return


    def get_data(self):
        """
        To use less memory the initial unscaled data is not stored in this class, so this function reverts the scaled data back and returns the initial unscaled data.
        """

        if self.testing:
            X_valst = self.X_test.copy()
            y_valst = self.y_test.copy()
        else:
            X_valst = self.X_val.copy()
            y_valst = self.y_val.copy()
        X_ls = dh.inverse_affine_scaling(
            [self.X_train.copy(), X_valst],
            scaler_obj=self.input_scaler_obj, is_input=True)
        X_train, X_valst = X_ls[0], X_ls[1]
        y_ls = dh.inverse_affine_scaling(
            [self.y_train.copy(), y_valst],
            scaler_obj=self.output_scaler_obj, is_input=False)
        y_train, y_valst = y_ls[0], y_ls[1]
        if self.log_scaling:
            tn = self.target_name[0]
            y_ls_0 = dh.inverse_log_scaling([y_train, y_valst])
            y_train, y_valst = y_ls_0[0][tn].values, y_ls_0[1][tn].values
        return X_train, pd.DataFrame(y_train, index=X_train.index, columns=[self.target_name]), X_valst, pd.DataFrame(
            y_valst, index=X_valst.index, columns=[self.target_name])

    def get_y_test_and_mix(self):
        if any(elem is None for elem in [self.mix_val, self.mix_train, self.val]):
            if any(elem is None for elem in [self.mix_val, self.mix_train]):
                return self.val, None, None
            else:
                print("test/val or mix_df_test not available.")
        else:
            return self.val, self.mix_val, self.mix_train

    def get_predictions(self):
        if any(elem is None for elem in [self.pred, self.pred_samples, self.pred_train, self.pred_train_samples]):
            print("No predictions available")
        else:
            return self.pred, self.pred_samples, self.pred_train, self.pred_train_samples

    def get_uncertainties(self, inv_errors=True, log10=False, train=False):
        if inv_errors:
            if any(elem is None for elem in [self.sigma_mix, self.sigma_epis, self.sigma_aleo]):
                print("No uncertainties available")
            else:
                if train:
                    print("Unscaled uncertainties for train set currently not available")
                else:
                    mix, epis, aleo = self.sigma_mix, self.sigma_epis, self.sigma_aleo
                return mix, epis, aleo
        else:
            if any(elem is None for elem in
                   [self.y_sigma_mix_scaled, self.y_sigma_epis_scaled, self.y_sigma_aleo_scaled]):
                print("No uncertainties available")
            else:
                if train:
                    mix, epis, aleo = self.y_sigma_aleo_scaled_train, self.y_sigma_epis_scaled_train, self.y_sigma_mix_scaled_train
                else:
                    mix, epis, aleo = self.y_sigma_mix_scaled, self.y_sigma_epis_scaled, self.y_sigma_aleo_scaled
                if log10:
                    return np.vstack((mix * np.log10(np.exp(1)),
                                      mix * np.log10(np.exp(1)))), \
                           np.vstack((epis * np.log10(np.exp(1)),
                                      epis * np.log10(np.exp(1)))), \
                           np.vstack((aleo * np.log10(np.exp(1)),
                                        aleo * np.log10(np.exp(1))))
                else:
                    return np.vstack((mix, mix)), np.vstack(
                        (epis, epis)), \
                           np.vstack((aleo, aleo))

    @classmethod
    def formatter(cls, key, decimals_round):
        if isinstance(key, float):
            return round(key, decimals_round)
        else:
            return key

    def __model_results_to_df(self, metrics_dict, sort_label=None, metrics_keys=None, decimals_round=4):
        """
        Creates a dataframe of all metrics, hyperparameters and other final values for all the models/permutations. The dataframe is then used to create the csv file with the results

        :param metrics_dict: dict.
        The value of the metrics the program has stored
        :param sort_label:
        Which metric the dataframe and thus final csv file should be sorted by.
        :param metrics_keys:
        The names of the metrics the user passed in to Initialization
        :param decimals_round:
        To what decimal point the metrics should be stored with in the csv.

        """

        params_permutations = pd.DataFrame(self.permutations_dicts)
        for k in list(metrics_dict):
            if len(metrics_dict[k]) == 0:
                del metrics_dict[k]
        metrics = pd.DataFrame(metrics_dict)
        for c in metrics_keys:
            if c not in metrics.columns:
                metrics_keys.remove(c)
        if metrics_keys is not None:
            metrics = metrics.reindex(columns=metrics_keys)
        # Round results
        for c in metrics.columns:
            metrics[c] = metrics[c].apply(lambda key: self.formatter(key, decimals_round))
        results_df = pd.concat([params_permutations, metrics], axis=1)
        if not self.testing:
            results_df = results_df.sort_values(by=sort_label)
            sort_label_old = sort_label
            sort_label += '*'
            results_df = results_df.rename(columns={sort_label_old: sort_label})
        results_df.index = np.arange(1, len(results_df) + 1)
        # Make layers column ready for csv
        if 'layers' in results_df.columns:
            results_df['layers'] = results_df['layers'].apply(lambda key: '-'.join([str(i) for i in key]))
        # Making empty cells empty and not nan
        results_df = results_df.replace(np.nan, '', regex=True)
        # removing zeros from scientific notation
        results_df = results_df.astype(object)
        if self.write_to_csv:
            main_dir = os.path.abspath(os.path.join(__file__, "../.."))
            dirname = main_dir + '/multiple_runs_csv_files'
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            runs = str(len(self.permutations_dicts))
            mutable_params = str(len(self.permutations_dicts[0]))
            if self.save_name is None:
                save_name = dirname + '/runs:' + runs + '_params:' + mutable_params
                if metrics_keys == self.training_metrics_keys:
                    save_name += '_training'
                elif metrics_keys == self.final_metrics_keys:
                    save_name += '_final'
                elif metrics_keys == self.testing_metrics_keys:
                    save_name += '_testing'
            else:
                save_name = dirname + '/' + self.save_name
                if metrics_keys == self.training_metrics_keys:
                    save_name += '_training'
                elif metrics_keys == self.final_metrics_keys:
                    save_name += '_final'
                elif metrics_keys == self.testing_metrics_keys:
                    save_name += '_testing'

            if self.job_array_bool:
                save_name += '_' + str(self.idx) + '.csv'
                results_df.index = np.arange(self.start + 1, self.end + 1)
                if self.idx == 0:
                    header = True
                else:
                    header = False
                results_df.to_csv(save_name, sep='\t', encoding='utf-8', float_format='%f', header=header)

            else:
                save_name += '.csv'
                if not self.overwrite:
                    save_name = hf.add_num_to_txt_name(save_name)
                results_df.to_csv(save_name, sep='\t', encoding='utf-8', float_format='%f')

            print("CSV-file saved as: ", save_name)
        print("\n Results: ")
        if self.latex_print:
            print(results_df.to_latex(index=False))
        else:
            print(results_df.to_string())
        print("\n")

    def __save_and_print(self, save_model=True):
        """
        Runs the save_model function and __model_results_to_df function.
        For the latter it also prepares all the hyperparameter configurations in permutations_dicts for printing and csv storing in __model_results_to_df,
        as permutation_dicts is changed during the course of the program.

        :param save_model:
        """

        stored_dist_dicts = []
        if self.permutations_dicts[0].get('distributions') is not None:
            for i, d in enumerate(self.permutations_dicts):
                new_dist_dict = dcth.printable_distributions_dict(d, d['distributions'][0]['prior'].dist,
                                                                  d['distributions'][1]['pmf'].dist)
                if not save_model:
                    del d['distributions']
                    self.permutations_dicts[i] = {**d, **new_dist_dict}
                else:
                    stored_dist_dicts.append(new_dist_dict)

        if save_model:
            if self.save_best_model:
                self.save_model(save_best=True)
            else:
                self.save_model(save_best=False)
            if self.permutations_dicts[0].get('distributions') is not None:
                for i, d in enumerate(self.permutations_dicts):
                    del d['distributions']
                    self.permutations_dicts[i] = {**d, **stored_dist_dicts[i]}

        if self.metrics_report == 'final':
            self.__model_results_to_df(sort_label=self.optim + '(scaled)', metrics_dict=self.final_metrics_dict,
                                       metrics_keys=self.final_metrics_keys)
        elif self.metrics_report == 'training':
            self.__model_results_to_df(sort_label=self.optim + '(scaled)', metrics_dict=self.final_metrics_dict,
                                       metrics_keys=self.final_metrics_keys)
            self.__model_results_to_df(sort_label=self.optim, metrics_dict=self.training_metrics_dict,
                                       metrics_keys=self.training_metrics_keys)
        elif self.metrics_report == 'testing':
            self.__model_results_to_df(metrics_dict=self.testing_metrics_dict,
                                       metrics_keys=self.testing_metrics_keys)
        else:
            print("Choose between: 'final', 'training'")

    @classmethod
    def __negloglik(cls, y, rv_y):
        negloglik = -tf.reduce_mean(rv_y.log_prob(tf.cast(y, tf.float64)))
        return negloglik

    def save_model_params_only(self):
        """
        Storing all model hyperparameters in "model folders", but not the weights (the model itself). This is convenient when one runs main_training.py but is intending to retrain with optimal
        parameters on a larger data set with main_testing.py. One can choose the best model from the metrics csv, and choose to load the corresponding "model folder" in main_testing.py.
        This is essentially a preparation function, and uses the function save_params() for the actual saving.

        """

        if hasattr(self.prior, 'nn_best_epoch'):
            nn_best_epoch = self.prior.best_epoch
        elif hasattr(self.pmf, 'nn_best_epoch'):
            nn_best_epoch = self.prior.best_epoch
        else:
            nn_best_epoch = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_dir = os.path.abspath(os.path.join(__file__, "../.."))
        dirname_params = main_dir + '/saved_parameters'
        if not os.path.exists(dirname_params):
            os.mkdir(dirname_params)
        elif self.perm_iteration == 0 and self.overwrite:
            shutil.rmtree(dirname_params)
            dirname_params = hf.add_num_to_txt_name(dirname_params, '')
            os.mkdir(dirname_params)
        elif self.perm_iteration == 0 and not self.overwrite:
            dirname_params = hf.add_num_to_txt_name(dirname_params, '')
            os.mkdir(dirname_params)

        if self.job_array_bool:
            dirname_params += '/model_' + str(
                self.perm_iteration + self.idx * len(self.permutations_dicts) + 1) + '_params'
        else:
            dirname_params += '/model_' + str(self.perm_iteration + 1) + '_params'
        if not os.path.exists(dirname_params):
            os.mkdir(dirname_params)

        os.getcwd()
        os.chdir(dirname_params)

        filename = 'model'
        model, params_dict, history = self.model, self.permutations_dicts[self.perm_iteration], self.history_save
        self.save_params(params_dict, filename, history, nn_best_epoch, self.bnn_best_epoch)
        os.chdir(current_dir)

    def save_model(self, save_best=True, dirname_bnn=None):
        """
        Saves model weights and hyperparameters.

        :param save_best: bool
        If set to true it will save the best run from multiple runs. If false it will save the current self.model.
        :param dirname_bnn: string
        Directory that the saved model should be saved to within the directory /saved_bnns
        """

        if dirname_bnn is None:
            if self.save_name is None:
                dirname_bnn = 'model_example'
            else:
                dirname_bnn = self.save_name

        if self.job_array_bool:
            dirname_bnn += '_' + str(self.idx)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_dir = os.path.abspath(os.path.join(__file__, "../.."))
        dirname_all_bnn = main_dir + '/saved_bnns'
        if not os.path.exists(dirname_all_bnn):
            os.mkdir(dirname_all_bnn)
        dirname_bnn = dirname_all_bnn + '/' + dirname_bnn
        if not self.overwrite:
            dirname_bnn = hf.add_num_to_txt_name(dirname_bnn, '')
        filename = 'model'

        if not os.path.exists(dirname_bnn):
            os.mkdir(dirname_bnn)

        os.getcwd()
        os.chdir(dirname_bnn)
        # filename_params = filename + '_params'
        if save_best:
            model, params_dict, history, mle, nn_best_epoch, bnn_best_epoch = self.current_best_model[0], \
                                                                              self.current_best_model[1], \
                                                                              self.current_best_model[2], \
                                                                              self.current_best_model[3], \
                                                                              self.current_best_model[4], \
                                                                              self.current_best_model[5]
        else:
            if hasattr(self.prior, 'mle'):
                mle = self.prior.mle.copy()
                nn_best_epoch = self.prior.best_epoch
            elif hasattr(self.pmf, 'mle'):
                mle = self.prior.mle.copy()
                nn_best_epoch = self.prior.best_epoch
            else:
                mle = None
                nn_best_epoch = None

            model, params_dict, history = self.model, self.permutations_dicts[self.perm_iteration], self.history_save

        self.save_params(params_dict, filename, history, nn_best_epoch, self.bnn_best_epoch, mle)

        filename_weights = filename + '_weights'
        model.save_weights(filename_weights)

        os.chdir(current_dir)
        print("\nModel saved at %s" % dirname_bnn)
        print("")

    def save_params(self, params_dict, filename, history, nn_best_epoch=None, bnn_best_epoch=None, mle=None):
        """
        Save parameters into a folder

        :param params_dict: dict.
        :param filename: string.
        The path of the folder (excuse the wrong name)
        :param history: dict.
        :param nn_best_epoch: int.
        :param bnn_best_epoch: int.
        :param mle: np.array
        The weights from the deterministic neural network used for initialization.

        """
        params_dict_copy = {x: params_dict[x] for x in params_dict if x not in ['distributions']}
        params_dict_copy['distributions'] = (
            {x: params_dict['distributions'][0][x] for x in params_dict['distributions'][0] if x not in ['prior']},
            {x: params_dict['distributions'][1][x] for x in params_dict['distributions'][1] if x not in ['pmf']})
        if params_dict.get('distributions') is not None:
            filename_distributions = filename + '_distributions'
            prior_init, pmf_init = params_dict['distributions'][0]['prior'].initializer_type, \
                                   params_dict['distributions'][1]['pmf'].initializer_type
            self.prior, self.pmf = params_dict['distributions'][0]['prior'], params_dict['distributions'][1]['pmf']
            params_dict_copy['distributions'][0]['prior'] = str(prior_init).split(' ')[2]
            params_dict_copy['distributions'][1]['pmf'] = str(pmf_init).split(' ')[2]
            with open(filename_distributions, "wb") as f:
                pickle.dump((str(prior_init).split(' ')[2], str(pmf_init).split(' ')[2]), f)

            filename_nn_history = filename + '_nn_history'
            if hasattr(self.prior, 'history'):
                history_nn = self.prior.history
                with open(filename_nn_history, "wb") as f:
                    pickle.dump(history_nn, f)
            elif hasattr(self.pmf, 'history'):
                history_nn = self.pmf.history
                with open(filename_nn_history, "wb") as f:
                    pickle.dump(history_nn, f)

        filename_params = filename + '_params'
        with open(filename_params, "wb") as f:
            pickle.dump((params_dict_copy, self.memory_usage_ls, mle, nn_best_epoch, bnn_best_epoch), f)
        filename_history = filename + '_history'
        with open(filename_history, "wb") as f:
            pickle.dump(history, f)

        filename_history_table = filename + '_history_table'
        pd.DataFrame(history).to_csv(filename_history_table, index=False)

    def load_model(self, dirname_bnn=None, write_to_csv=False, overwrite=False, sample_size=None, load_only=False):
        """
        Used to load model, that is, a trained model.
        """

        if not self.testing and list(self.training_metrics_dict.values())[0]:
            self.__reset()
        self.write_to_csv = write_to_csv
        self.overwrite = overwrite
        self.store_scaled_errors = True

        if dirname_bnn is None:
            print("Set directory path")
            return
        if os.path.exists(dirname_bnn):
            print("Loading stored model at %s" % dirname_bnn)
            permutation_dict = self.load_parameters(dirname_bnn, sample_size)

            if not load_only:
                self.filename_history = dirname_bnn + '/model_history'
                self.filename_weights = dirname_bnn + '/model_weights'
                self.run(**permutation_dict, loaded_model=True)
                self.__save_and_print(save_model=False)
        else:
            raise FileNotFoundError("Can't find model at %s" % dirname_bnn)

    def load_parameters(self, dirname_bnn, sample_size=None):
        """
        Loading parameters from folder. Can be loaded from both save_model() type folders and save_model_params_only folders.
        Convenient when using main_testing.py av main_training.py

        :param dirname_bnn: string.
        folder name
        :param sample_size: int.

        """
        if dirname_bnn is None:
            print("Set directory path")
            return
        if os.path.exists(dirname_bnn):
            print("Loading parameters from %s" % dirname_bnn)

            # Loading parameters
            filename_params = dirname_bnn + '/model_params'
            if os.path.exists(filename_params):
                with open(filename_params, "rb") as f:
                    permutation_dict, self.memory_usage_ls, mle, nn_best_epoch, bnn_best_epoch = pickle.load(
                        f)
                filename_distributions = dirname_bnn + '/model_distributions'
                if os.path.exists(filename_distributions):
                    with open(filename_distributions, "rb") as f:
                        prior_init, pmf_init = pickle.load(
                            f)
                    if not str(self.pmf.initializer_type).split(' ')[2] == pmf_init:
                        print("WARNING: %s is not the correct pmf-initializer for the loaded model. Use %s" % (
                            str(self.pmf.initializer_type).split(' ')[2], pmf_init))
                    else:
                        if pmf_init.split('_')[1] == 'nn':
                            self.pmf.mle = mle
                            self.pmf.epochs = permutation_dict['epochs']
                            self.pmf.patience = permutation_dict['patience']

                    if not str(self.prior.initializer_type).split(' ')[2] == prior_init:
                        print("WARNING: %s is not the correct prior-initializer for the loaded model. Use %s" % (
                            str(self.prior.initializer_type).split(' ')[2], prior_init))
                    else:
                        if prior_init.split('_')[1] == 'nn':
                            self.prior.mle = mle
                            self.prior.epochs = permutation_dict['epochs']
                            self.prior.patience = permutation_dict['patience']

                    permutation_dict['distributions'][0]['prior'] = self.prior
                    permutation_dict['distributions'][1]['pmf'] = self.pmf
                else:
                    raise FileNotFoundError(
                        "Could not find parameters at %s. Searching for '/model_distributions'" % filename_distributions)

            else:
                raise FileNotFoundError(
                    "Could not find parameters at %s. Searching for /model_params" % filename_params)
            if sample_size is not None:
                permutation_dict['sample_size'] = sample_size
            self.permutations_dicts = [permutation_dict]
            # self.bnn_best_epoch = bnn_best_epoch
            self.loaded_parameters = True
            return permutation_dict
        else:
            raise FileNotFoundError(
                "Could not find folder at %s" % dirname_bnn)
            return

    def __load_weights(self):
        """Used to load model weight, used by other functions."""

        if os.path.exists(self.filename_history):
            with open(self.filename_history, "rb") as f:
                history = pickle.load(
                    f)
        else:
            raise FileNotFoundError(
                "Could not find history at %s. Searching for /model_history" % self.filename_history)

        if os.path.exists(self.filename_weights + '.index'):
            self.model.load_weights(self.filename_weights)
        else:
            raise FileNotFoundError(
                "Could not find weights at %s. Searching for /model_weights" % self.filename_weights)
        print("\nModel has been successfully loaded from disk")
        return history

    def __reset(self):
        """Resets some dictionaries and lists to what they were initially before hyperparameter tuning."""
        self.training_metrics_dict, self.final_metrics_dict, self.training_metrics_keys, self.final_metrics_keys = dcth.metrics_filter(
            self.monitored_metrics, self.metrics_report, optim=self.optim)
