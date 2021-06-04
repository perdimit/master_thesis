import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from bnn_module import Initialization as init
from bnn_module import DistributionInitialization as dist_init

seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)

mirrored_strategy = False
dtype = "float64"
# dataset_train_name = 'EWonly_PMCSX_22-22_train'
# dataset_test_name = 'EWonly_PMCSX_22-22_test'
# target = ["1000022_1000022_13000_NLO_1"]

parameter_path_name = 'saved_bnns/model_training_1'
dataset_train_name = 'EWonly_PMCSXUV_23-24_23--24_22-22_23-37_35-24_25-24_25--24_22--37_train'
dataset_test_name = 'EWonly_PMCSXUV_23-24_23--24_22-22_23-37_35-24_25-24_25--24_22--37_test'
target = ["1000023_1000024_13000_NLO_1"]

hpc_bool = os.getenv('ON_HPC', 'False') == 'True'
if hpc_bool:
    dataset_train_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'], dataset_train_name)
    dataset_test_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'], dataset_test_name)
    parameter_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'], parameter_path_name)
    if mirrored_strategy:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
        BATCH_SIZE_PER_REPLICA = 32
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    else:
        strategy = None
        BATCH_SIZE = 32
else:
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset_train_path = dir_path + '/Data Harvest/' + dataset_train_name
    dataset_test_path = dir_path + '/Data Harvest/' + dataset_test_name
    strategy = None
    manual_job_array = None

df_train = pd.read_csv(dataset_train_path, sep="\t", skipinitialspace=True, index_col=0)
df_test = pd.read_csv(dataset_test_path, sep="\t", skipinitialspace=True, index_col=0)
# df_train = df_train.drop(columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
# df_test = df_test.drop(columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
df_train = df_train[["1000023_1000024_13000_NLO_1", 'm1000023', 'm1000024', 'nmix21', 'nmix22', 'nmix23', 'nmix24', 'vmix11', 'vmix12']]
df_test = df_test[["1000023_1000024_13000_NLO_1", 'm1000023', 'm1000024', 'nmix21', 'nmix22', 'nmix23', 'nmix24', 'vmix11', 'vmix12']]
mix_fix = False
features_len = len(df_train.columns) - len(target)

############## Declaring priors and posterior mean field distributions ##############
gaussian_prior = dist_init.SetDistribution(dtype=dtype, trainable=False, dist='Laplace')
gaussian_prior.set_constant_initializer()

gaussian_pmf = dist_init.SetDistribution(dtype=dtype, trainable=True, dist='Laplace')
gaussian_pmf.set_random_gaussian_initializer()

# mix_prior = dist_init.SetMixtureDistribution(dtype=dtype, trainable=False)
# mix_prior.set_constant_mixture_initializer()
#
# mix_pmf = dist_init.SetMixtureDistribution(dtype=dtype, trainable=True)
# mix_pmf.set_constant_mixture_initializer()

nn_prior = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Laplace')
nn_prior.set_nn_constant_initializer()

nn_pmf = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior, dist='Laplace')
nn_pmf.set_nn_complex_initializer()

nn_pmf_g = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior, dist='Gaussian')
nn_pmf_g.set_nn_complex_initializer()

metrics = ['train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'train_negloglik', 'test_negloglik', 'test_elbo',
           'test_loss', 'mix_res_stddev', 'mix_res_mean', 'caos', 'caus', 'kl_elbo', 'stopped_epoch', 'best_epoch']

model = init.Initialization(data=df_train, data_test=df_test, target_name=target, prior=nn_prior, pmf=nn_pmf,
                            seed=seed, log_scaling=True, affine_scaler='MinMaxScaler', affine_target_scaling=True,
                            monitored_metrics=metrics, metrics_report='testing', optim_by_unregularized_loss=True, model_name='testing',
                            strategy=strategy, mean_bool=False, mix_fix=mix_fix)

# current_dir = os.path.dirname(os.path.abspath(__file__))
# model.test_run_with_params('/home/per-dimitri/Dropbox/Master/BayesianNeuralNetwork/saved_csv/lae_false/saved_bnns/model_training_11', sample_size=2000)
# path_model = '/home/per-dimitri/Dropbox/Master/BayesianNeuralNetwork/saved_csv/2324/saved_bnns/model_training_1'
model.test_run_with_params(parameter_path, sample_size=2000, save_model=True)

