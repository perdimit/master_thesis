import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import datetime
from bnn_module import Initialization as init
from bnn_module import DistributionInitialization as dist_init

""" This script is used for finding the optimal model. It let's set various hyperparameters and distributions, and train a BNN for every configuration.
The performance measure of all the various models can be printed out in a csv. 
The script runs the Initialization class which splits the passed-in-data set into three sets: training, early stopping validation, and model validation. 
The purpose of this is to find the model that performs the best on the model validation set (or on all three). 
The model parameters of the chosen model can be loaded in the main_testing.py file which trains a BNN but without a dedicated model validation set (such that the training set is larger),
and reports final performance measures on a final test set (should be made initially by main_data_harvest.py)."""

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Set to true to use multiple GPUs on the same node (Max 4 on SAGA). Only necessary with large data sets and deep networks.
mirrored_strategy = False

# Code requires float64, float32 gives floating point error and division by zero.
dtype = "float64"

# filename of dataset
dataset_name = 'EWonly_PMCSXUV_23-24_23--24_22-22_23-37_35-24_25-24_25--24_22--37_train'

# name of target column, i.e. the cross section.
# target = ["1000022_1000022_13000_NLO_1"]
target = ["1000023_1000024_13000_NLO_1"]

# When running on HPC such as SAGA the job file must contain the environment variable ON_HPC
hpc_bool = os.getenv('ON_HPC', 'False') == 'True'
if hpc_bool:
    # When running on HPC:
    dataset_path = os.path.join(os.environ['SLURM_SUBMIT_DIR'], dataset_name)
    if mirrored_strategy:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {strategy.num_replicas_in_sync}")
        BATCH_SIZE_PER_REPLICA = 32
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    else:
        strategy = None
        BATCH_SIZE = 32

    # Manually set the jobs distribution. If you have 32 models being run by 16 jobs
    # then the uniform solution is the default if manual_job_array is None, that is, models/jobs = 2 models per script.
    # But if you know the last 8 models are deeper networks, and thus require more time,
    # you can for example use manual_job_array = [3]*8 + [1]*8 instead. So 8 jobs do 3 models each, and 8 jobs 1 model each.
    manual_job_array = None
    if manual_job_array is not None:
        print("Number of jobs set to: ", sum(manual_job_array))
else:
    # When running on private computer
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset_path = dir_path + '/Data Harvest/' + dataset_name
    strategy = None
    BATCH_SIZE = 32
    manual_job_array = None

""" In the next lines of code the relevant columns are selected from the data file. mix_fix only works with "1000022_1000022_13000_NLO_1" case at the moment. It calculates whether a given cross section
value corresponds to masses that are mostly bino, wino or higgsino."""
df = pd.read_csv(dataset_path, sep="\t", skipinitialspace=True, index_col=0)
df = df[["1000023_1000024_13000_NLO_1", 'm1000023', 'm1000024', 'nmix21', 'nmix22', 'nmix23', 'nmix24', 'vmix11', 'vmix12']]
mix_fix = False

"""Use these lines to pick a random subset of the data. Good for testing of code as network trains faster with less data"""
# drop_indices = np.random.choice(df.index, 2000, replace=False)
# df = df.drop(drop_indices)

features_len = len(df.columns) - len(target)

""" *** Declaring priors and posterior mean field distributions (pmf) ***
 Set the priors and pmfs you want to use. Multiple distributions can be initialized in order to test many in one hyperparameter tuning run.

In the following example a gaussian prior and laplace prior is set, and a Gaussian pmf.
The prior should be set to non-trainable for the most Bayesian approach.
 
gaussian_prior = dist_init.SetDistribution(dtype=dtype, trainable=False, dist='Gaussian')
gaussian_prior.set_constant_initializer()

laplace_prior = dist_init.SetDistribution(dtype=dtype, trainable=False, dist='Laplace')
laplace_prior.set_constant_initializer()

gaussian_pmf = dist_init.SetDistribution(dtype=dtype, trainable=True, dist='Gaussian')
gaussian_pmf.set_random_gaussian_initializer()

These can be used later on by declaring
distributions = [[gaussian_prior, gaussian_pmf], [laplace_prior, gaussian_pmf]]
such that it will train both a network with laplace and gaussian priors.
Other types of distributions can be made by the user by creating similar classes as those set in dist_init, that is, in DistributionInitialization.py.

***Deterministic Pretraining***
Instead of using a dist_init.SetDistribution the dist_init.SetNNDistribution class is recommended, which will give a more empirical bayes approach.
This trains a deterministic neural network first in order to set the BNN posterior and prior initial values.
This is necessary for good results. With no pretraining the BNN will underfit.
Note: Unlike SetDistribution, for the SetNNDistribution class an object cannot be reused. Meaning that one must set e.g.
distributions = [[gaussian_prior, gaussian_pmf_1], [laplace_prior, gaussian_pmf_2]].

To avoid training two deterministic neural network separately for the prior and the pmf initial values, you can link the prior and pmf objects, such that you train one neural network.
E.g.
nn_prior_laplace = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Laplace')
nn_pmf_gauss = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior_laplace, dist='Gaussian')

 """

nn_prior_laplace = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Laplace')
nn_prior_laplace.set_nn_constant_initializer()
nn_prior_gauss = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Gaussian')
nn_prior_gauss.set_nn_constant_initializer()

nn_pmf_gauss_1 = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior_gauss, dist='Gaussian')
nn_pmf_gauss_1.set_nn_complex_initializer()
nn_pmf_gauss_2 = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior_laplace, dist='Gaussian')
nn_pmf_gauss_2.set_nn_complex_initializer()

"""
*** METRICS ***

train: Metrics on the targets used for training.
es_val: Metrics on the validation set used for early stopping.
model_val: Metrics on the validation set used to compare model and pick the best one. Arguably the most interesting metrics.

elbo: the variational free energy, or the negative expected lower bound
kl_elbo and negloglik: the elbo consist of a sum of a kl term and a negative log likelihood term, specifically kl_elbo + negloglik.
The kl_elbo term is the "bayesian" term (complexity cost) and reflects the fitting of the weights to the prior (the smaller the closer),
while the negloglik is the "data" term (likelihood cost), fitting the network to the data.
r2 and rmse: standard regression metrics

mix_res_stddev: This is the standard deviation of the standardized residuals distribution. If standardized residuals
behave like a normal dist then smaller then a value smaller then 1 means the BNN uncertainty underestimates the true uncertainty (what we want).
Above 1 means it overestimates. This is a rough estimate depending on the above assumption.
See chapter 6.4 in thesis for futher explanation.

mix_res_mean: mixture_residual_mean. This is the mean of the standardized residuals distribution using the gaussian mixture standard deviation (mix sd).
If zero then mix sd neither systematically overestimates nor underestimates the true residual error.
See chapter 6.4 in thesis for futher explanation.

aleo and epis is the aleoteric and epistemic components of the gaussian mixture standard deviation.
Using only the "mix" metric is sufficient.
See chapter 5.7.2

caos and caus is "coverage area over/under score". Both are between 0 and 1. The higher the caos value, the more the network overestimates the true uncertainty,
and the higher the caus value, the more the network underestimates the true uncertainty.
See chapter 6.4.

stopped_epoch: the epoch the network stopped training on.
best_epoch: the epoch with the weight chosen.
note that patience = stopped_epoch - best_epoch 

nn_time : time used training determinsitic neural network
bnn_time : time used training Bayesian neural network
pred_time : time used for one round of predictions.
"""

metrics = ['model_val_elbo', 'train_rmse', 'es_val_rmse', 'model_val_rmse', 'train_r2', 'es_val_r2', 'model_val_r2',
           'model_val_negloglik', 'kl_elbo', 'mix_res_stddev',
           'mix_res_mean', 'epis_res_stddev', 'epis_res_mean', 'aleo_res_stddev', 'aleo_res_mean', 'caos', 'caus',
           'stopped_epoch', 'best_epoch', 'nn_time', 'bnn_time', 'pred_time']


"""
See initialization class for explanation of the various inputs in model = init.Initialization(...)
"""

model = init.Initialization(df, target_name=target,
                            test_fracs=[0.1875, 0.1875],
                            seed=seed, log_scaling=True, affine_scaler='MinMaxScaler', affine_target_scaling=True,
                            monitored_metrics=metrics, metrics_report='training', optim_by_unregularized_loss=True,
                            model_name='model_training',
                            strategy=strategy, manual_job_array=manual_job_array, mean_bool=False,
                            sort_permutations_by='layers', latex_print=False, mix_fix=mix_fix)


"""***Hyperparameters and model selection***

After having declared model = init.Initialization(...)
Use the multiple_runs function to pass in all the various hyperparameters, distributions etc you would like to check:

model.multiple_runs(param_dict=params, write_to_csv=True, overwrite=True,
                                    save_best_model=True, save_all_model_hyperparams=True)
                                    
With write_to_csv=True a csv will be written giving all the Metrics set in the Initialization object for every hyperparameter configuration.
save_best_model=True: Saves to file the weights and hyperparameters of the best model (Chosen by lowest ELBO as default). This can be loaded again by using 
model.load_model('filename', sample_size=sample_size).
save_all_model_hyperparams=True: Saves to file the hyperparameters of all models. This is convenient as in main_testing.py one may simply give the file path of
the stored hyperparameters to run the final training. See main_testing.py for the purpose of this.

The multiple_runs function takes in hyperparameters configurations in the form of a dictionary of lists:
E.g. 
params = {'epochs': [2000, 4000], 'patience': [100, 10000], 'layers': [[features_len, 20, 20, 20, 20, 20, 1], [features_len, 20, 20, 20, 20, 20, 20, 1]], 'learning_rate': [0.1, 0.01]}
All permutations of these inputs will be trained and reported. So that is 16 training runs in this example.
It is highly recommended using HPC for this with a jobs array. Using e.g. 64 CPUs, giving 4 CPUs to each permutation and running them simultaneously (depending on the time you have).

***********************************
*** hyperparameters description ***
***********************************

Parameter dictionary key names in ''

'sample_size': After a BNN has been trained it will test the model on a validation data set. This is number of forward passes that will be made for each cross section prediction.
'epochs': Number of epochs
'patience': How epochs to run for after a new low (in terms of ELBO) has been found.
'layers': Layers matrix. The different layer combinations to try. This is a list of lists. Note that the output layer must be equal to 1, however, 
it will yield two outputs, the standard deviation and the mean.
'batch_size': The number for training datapoints in one mini-batch. Ideal batch size depends largely on the size of the data set.
'use_flipout': Turns on flipout. Only works without pretraining and with tensorflow probablity's standard Gaussian initialization.
'learning_rate': The learning rate of gradient descent, specifically the ADAM optimizer.
'activation': Which activation function to use. swish or mish is recommended.
'kl_weight': The weight of the kl_elbo term. If nothing is passed in this will be equal to 1/(number of batches). 
If annealing is used this parameter has a different purpose. See annealing below.

*********************************
*** Passing in priors and pmf ***
*********************************

The priors and pmfs are passed in together with the hyperparameters, however they must be passed in as follows

1.  Define objects as previously described:
nn_prior_laplace = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Laplace')
nn_pmf_gauss = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior_laplace, dist='Gaussian')

2. Make dictionaries for priors and pmfs, respectively, and pass in the object as follow.
nn_prior_par_laplace = {'prior': nn_prior_laplace, 'scale': [0.00015, 0.0015], 'learning_rate': [0.001, 0.0001]}
nn_pmf_par_gauss = {'pmf': nn_pmf_gauss, 'delta': [1e-6, 1e-5]}

The hyperparameters are described in the relevant class in DistributionInitialization.py. Using the previous example where we had 16 permutations, passing in this pmf and prior
we would now have 16*2*2*2 = 128 permutations that would be tested.
Note that the learning rate here is for pretraining, i.e. the DNN learning rate,
while passing learning_rate directly into params is for the BNN.

3. Make
distributions = [[nn_prior_par_laplace, nn_pmf_par_gauss]]
and add to params
params = {'epochs': [2000, 4000], 'patience': [100, 10000], 'layers': [[features_len, 20, 20, 20, 20, 20, 1], [features_len, 20, 20, 20, 20, 20, 20, 1]], 'learning_rate': [0.1, 0.01]
          'distributions': distributions}
          
****************************
*** Annealing parameters ***
****************************

Three different annealing schemes are available.
The annealing methods seek to tune the trade off in the ELBO cost function between the kl_elbo and negloglik throughout a training session.
The three avaliable are
Linear epoch annealing: 'linear'
Sinusoidal epoch annealing: 'sine'
Exponential batch annealing: 'exp'
No annealing: 'None'
Use by passing to params list e.g. 'annealing_type' = ['linear']

The 'linear' and 'sine' modify the kl_elbo term over epochs, the former steadly increases the magnitude of the kl_elbo term, while the latter varies it over the annealing phase.
The 'exp' modify the kl_elbo term over mini-batches (within each epoch) and restarts for every epoch.

Hyperparameters:

***'linear'***
'annealing': Epoch to start annealing
'kl_weight': Starting kl weight
'max_kl_weight': Final kl_weight
'annealing_phase': For how many epochs the annealing should go over.

***'sine'***
Same as linear but with one extra parameter
'periods': number of sine periods between start epoch and end epoch.

***'exp'***
'kl_weight': starting kl_weight
'kappa': the coefficient exp(-minibatch * kappa). It controls the rate of decrease over batches. 

See 5.6.3 in thesis for explanation of linear and exponential annealing.

See chapter 7 in thesis for discussion on best configurations.


"""

sample_size = [2000]

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


distributions = [[nn_prior_par_gauss, nn_pmf_par_gauss_1], [nn_prior_par_laplace, nn_pmf_par_gauss_2]]


params = {'epochs': [epochs], 'patience': [patience], 'layers': layers_mat, 'batch_size': batch_sizes,
          'activation': activation, 'sample_size': sample_size,
          'distributions': distributions, 'learning_rate': learning_rate,
          'annealing_type': at, 'kappa': kappa}

t = time.time()
perm, results = model.multiple_runs(param_dict=params, write_to_csv=True, overwrite=True,
                                    save_best_model=True, save_all_model_hyperparams=True)

""" To load model an already trained model, use the following lines instead of model.multiple_runs. Make sure you set the right file path."""
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model.load_model(current_dir + '/saved_bnns/model_training', sample_size=sample_size[0])

pred_time = time.time() - t
print("Total runtime: ", str(datetime.timedelta(seconds=pred_time)))
