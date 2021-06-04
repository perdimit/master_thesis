import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import time
import datetime
from bnn_module import Initialization as init
from bnn_module import DistributionInitialization as dist_init


seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)

mirrored_strategy = False
dtype = "float64"
# dataset_name = 'EWonly_PMCSX_22-22_train'
dataset_name = 'EWonly_PMCSXUV_23-24_23--24_22-22_23-37_35-24_25-24_25--24_22--37_train'
# target = ["1000022_1000022_13000_NLO_1"]
target = ["1000023_1000024_13000_NLO_1"]

# job file at HPC must contain environment variable ON_HPC
hpc_bool = os.getenv('ON_HPC', 'False') == 'True'
if hpc_bool:
    dataset_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'], dataset_name)
    if mirrored_strategy:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
        BATCH_SIZE_PER_REPLICA = 32
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    else:
        strategy = None
        BATCH_SIZE = 32

    # Manually set the jobs distribution. If you have 32 models being run by 16 scripts/jobs
    # then the uniform solution is the default if manual_job_array is None, that is, 2 models per script.
    # But if you know the last 8 models are deeper networks, and thus require more time,
    # you can for example use manual_job_array = [3]*8 + [1]*8 instead. So 8 scripts do 3 models each, and 8 scripts 1 model each.
    manual_job_array = None
    if manual_job_array is not None:
        print("Number of jobs set to: ", sum(manual_job_array))
else:
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset_path = dir_path + '/Data Harvest/' + dataset_name
    strategy = None
    BATCH_SIZE = 1024
    manual_job_array = None

df = pd.read_csv(dataset_path, sep="\t", skipinitialspace=True, index_col=0)
# df = pd.read_csv(dataset_path, sep=" ", skipinitialspace=True)
# df = df.iloc[:, :-1]
# df = df.drop(columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
df = df[["1000023_1000024_13000_NLO_1", 'm1000023', 'm1000024', 'nmix21', 'nmix22', 'nmix23', 'nmix24', 'vmix11', 'vmix12']]
mix_fix = False
print(df.columns)
# drop_indices = np.random.choice(df.index, 7000, replace=False)
# df = df.drop(drop_indices)
features_len = len(df.columns) - len(target)
print(features_len)

############## Declaring priors and posterior mean field distributions ##############
# gaussian_prior = dist_init.SetDistribution(dtype=dtype, trainable=False, dist='Gaussian')
# gaussian_prior.set_constant_initializer()
# #
# # laplace_prior = dist_init.SetDistribution(dtype=dtype, trainable=False, dist='Laplace')
# # laplace_prior.set_constant_initializer()
# #
# gaussian_pmf = dist_init.SetDistribution(dtype=dtype, trainable=True, dist='Gaussian')
# gaussian_pmf.set_random_gaussian_initializer()

# mix_prior = dist_init.SetMixtureDistribution(dtype=dtype, trainable=False)
# mix_prior.set_constant_mixture_initializer()
#
# mix_pmf = dist_init.SetMixtureDistribution(dtype=dtype, trainable=True)
# mix_pmf.set_constant_mixture_initializer()

# prior_1 = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Gaussian')
# prior_1.set_nn_constant_initializer()

# prior_2 = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Gaussian')
# prior_2.set_nn_constant_initializer()

# pmf_1 = dist_init.SetNNDistribution(dtype, trainable=True, link_object=prior_1, dist='Gaussian')
# pmf_1.set_nn_complex_initializer()

# pmf_2 = dist_init.SetNNDistribution(dtype, trainable=True, link_object=prior_2, dist='Gaussian')
# pmf_2.set_nn_complex_initializer()

nn_prior_laplace = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Laplace')
nn_prior_laplace.set_nn_constant_initializer()
nn_prior_gauss = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Gaussian')
nn_prior_gauss.set_nn_constant_initializer()

nn_pmf_gauss_1 = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior_gauss, dist='Gaussian')
nn_pmf_gauss_1.set_nn_complex_initializer()
nn_pmf_gauss_2 = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior_laplace, dist='Gaussian')
nn_pmf_gauss_2.set_nn_complex_initializer()

############## Delcaring a model run ###################

metrics = ['model_val_elbo', 'train_rmse', 'es_val_rmse', 'model_val_rmse', 'train_r2', 'es_val_r2', 'model_val_r2',
           'model_val_negloglik', 'kl_elbo', 'mix_res_stddev',
           'mix_res_mean', 'epis_res_stddev', 'epis_res_mean', 'aleo_res_stddev', 'aleo_res_mean', 'caos', 'caus',
           'stopped_epoch', 'best_epoch', 'nn_time', 'bnn_time', 'pred_time']

model = init.Initialization(df, target_name=target,
                            test_fracs=[0.1875, 0.1875],
                            seed=seed, log_scaling=True, affine_scaler='MinMaxScaler', affine_target_scaling=True,
                            monitored_metrics=metrics, metrics_report='training', optim_by_unregularized_loss=True,
                            model_name='model_training',
                            strategy=strategy, manual_job_array=manual_job_array, mean_bool=False,
                            sort_permutations_by='layers', latex_print=False, mix_fix=mix_fix)


sample_size = [2000]
# elu_alpha = [0.3]

if hpc_bool:
    epochs = 40000
    patience = 3000
    layers_mat = [[features_len, 20, 20, 20, 20, 20, 1], [features_len, 20, 20, 20, 20, 20, 20, 1], [features_len, 30, 30, 30, 30, 30, 1]]
    batch_sizes = [32]
    flipout_bools = [False]
    activation = ['swish']
    learning_rate = [0.0001]
    at = ['exp']
    kappa = [0.021, 0.071, 0.136, 0.361, 0.693, 0.96]
    nn_prior_par_gauss = {'prior': nn_prior_gauss, 'scale': [0.00015], 'learning_rate': [0.001]}
    nn_prior_par_laplace = {'prior': nn_prior_laplace, 'scale': [0.00015], 'learning_rate': [0.001]}
    nn_pmf_par_gauss_1 = {'pmf': nn_pmf_gauss_1, 'delta': [1e-6]}
    nn_pmf_par_gauss_2 = {'pmf': nn_pmf_gauss_2, 'delta': [1e-6]}

else:
    epochs = 40000
    patience = 3000
    layers_mat = [[features_len, 20, 20, 20, 20, 20, 1], [features_len, 20, 20, 20, 20, 20, 20, 1], [features_len, 30, 30, 30, 30, 30, 1]]
    batch_sizes = [32]
    flipout_bools = [False]
    activation = ['swish']
    learning_rate = [0.0001]
    at = ['exp']
    kappa = [0.021, 0.071, 0.136, 0.361, 0.693, 0.96]
    nn_prior_par_gauss = {'prior': nn_prior_gauss, 'scale': [0.00015], 'learning_rate': [0.001]}
    nn_prior_par_laplace = {'prior': nn_prior_laplace, 'scale': [0.00015], 'learning_rate': [0.001]}
    nn_pmf_par_gauss_1 = {'pmf': nn_pmf_gauss_1, 'delta': [1e-6]}
    nn_pmf_par_gauss_2 = {'pmf': nn_pmf_gauss_2, 'delta': [1e-6]}


# prior_1_par = {'prior': prior_1, 'scale': [0.8, 1, 1.2], 'learning_rate': [0.001], 'soft_coef': [0.01, 0.001, 0.0001]}
# prior_2_par = {'prior': prior_2, 'scale': [0.8, 1, 1.2], 'learning_rate': [0.001], 'soft_coef': [0.01, 0.001, 0.0001]}
# pmf_1_par = {'pmf': pmf_1, 'delta': [1e-5, 1e-6, 1e-7]}
# pmf_2_par = {'pmf': pmf_2, 'delta': [1e-5, 1e-6, 1e-7]}
# distributions = [[prior_1_par, pmf_1_par], [prior_2_par, pmf_2_par]]
# distributions = [[nn_prior_par_gauss, nn_pmf_par_gauss], [nn_prior_par_laplace, nn_pmf_par_laplace]]
distributions = [[nn_prior_par_gauss, nn_pmf_par_gauss_1], [nn_prior_par_laplace, nn_pmf_par_gauss_2]]
# distributions = [[nn_prior_par_gauss, nn_pmf_par_gauss_1]]
t = time.time()
# params = {'epochs': [epochs], 'patience': [patience], 'layers': layers_mat, 'batch_size': batch_sizes,
#           'activation': activation, 'use_flipout': flipout_bools, 'sample_size': sample_size,
#           'distributions': distributions, 'learning_rate': learning_rate,
#           'annealing': start_annealing, 'max_kl_weight': max_kl_weight, 'annealing_phase': annealing_phase, 'kl_weight': kl_weight, 'annealing_type': at, 'periods': periods}
params = {'epochs': [epochs], 'patience': [patience], 'layers': layers_mat, 'batch_size': batch_sizes,
          'activation': activation, 'sample_size': sample_size,
          'distributions': distributions, 'learning_rate': learning_rate,
          'annealing_type': at, 'kappa':kappa}

# 'distributions': distributions, 'learning_rate': learning_rate, 'annealing': start_annealing, 'max_kl_weight': max_kl_weight, 'annealing_phase': annealing_phase, 'kl_weight': kl_weight, 'annealing_type': at, 'soft_coef_ll': soft_coef_ll}
perm, results = model.multiple_runs(param_dict=params, write_to_csv=True, overwrite=True,
                                    save_best_model=True, save_all_model_hyperparams=True)

# print("\nRuntimes:", model.time_usage_ls)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model.load_model(current_dir + '/saved_bnns/model_training', sample_size=sample_size[0])
# model.load_model('/home/per-dimitri/Dropbox/Master/BayesianNeuralNetwork/saved_csv/Sine annealing runs/may15_savedmodel/model_training_2', sample_size=sample_size[0])
pred_time = time.time() - t
print("Total runtime: ", str(datetime.timedelta(seconds=pred_time)))
