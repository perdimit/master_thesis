import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import seaborn as sns
import prediction_plots as pp
import pickle
import time
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(1, path + '/Models')
# noinspection PyUnresolvedReferences
from bnn_module import Initialization as init
# noinspection PyUnresolvedReferences
from bnn_module import DistributionInitialization as dist_init
# noinspection PyUnresolvedReferences
from bnn_module import statistical_measures as sm
import os
import scipy.stats as stats

t = time.time()


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'DeJavu Serif',
    "font.serif": "Computer Modern Roman",
    "font.sans-serif": ['Computer Modern Roman']})

plt.rc('font', size=26)  # controls default text size
plt.rc('axes', titlesize=26)  # fontsize of the title
plt.rc('axes', labelsize=26)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=20)  # fontsize of the y tick labels
plt.rc('legend', fontsize=18)  # fontsize of the legend

colors = {'Fire Opal': '#EE6352', 'Emerald': '#59CD90', 'Verdigris': '#4CBAB3', 'Cerulean Crayola': '#3FA7D6',
          'Maximum Yellow Red': '#FAC05E',
          'Vivid Tangerine': '#F79D84', 'Deep Taupe': '#876A6C', 'Prussian Blue': '#173753'}

seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)

dtype = "float64"

remove_n = 1000


def set_data(testing, xsec='2222'):
    #

    if xsec == '2222':
        dataset_train_name = 'EWonly_PMCSX_22-22_train'
        dataset_test_name = 'EWonly_PMCSX_22-22_test'
        target = ["1000022_1000022_13000_NLO_1"]
    else:
        # parameter_path_name = 'saved_bnns/model_training_1'
        dataset_train_name = 'EWonly_PMCSXUV_23-24_23--24_22-22_23-37_35-24_25-24_25--24_22--37_train'
        dataset_test_name = 'EWonly_PMCSXUV_23-24_23--24_22-22_23-37_35-24_25-24_25--24_22--37_test'
        target = ["1000023_1000024_13000_NLO_1"]

    # df_test = df_test[["1000023_1000024_13000_NLO_1", 'm1000023', 'm1000024', 'nmix21', 'nmix22', 'nmix23', 'nmix24', 'vmix11', 'vmix12']]

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dataset_train_path = dir_path + '/Data Harvest/' + dataset_train_name
    dataset_test_path = dir_path + '/Data Harvest/' + dataset_test_name

    df_train = pd.read_csv(dataset_train_path, sep="\t", skipinitialspace=True, index_col=0)
    df_test = pd.read_csv(dataset_test_path, sep="\t", skipinitialspace=True, index_col=0)

    if xsec == '2222':
        df_train = df_train.drop(
            columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
    else:
        df_train = df_train[
            ["1000023_1000024_13000_NLO_1", 'm1000023', 'm1000024', 'nmix21', 'nmix22', 'nmix23', 'nmix24', 'vmix11',
             'vmix12']]

    if testing:
        if xsec == '2222':
            df_test = df_test.drop(
                columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
        else:
            df_test = df_test[
                ["1000023_1000024_13000_NLO_1", 'm1000023', 'm1000024', 'nmix21', 'nmix22', 'nmix23', 'nmix24',
                 'vmix11', 'vmix12']]
        metrics = ['train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'train_negloglik', 'test_negloglik', 'test_elbo',
                   'test_loss', 'mix_res_stddev', 'mix_res_mean', 'caos', 'caus', 'kl_elbo', 'stopped_epoch',
                   'best_epoch']
        print("Length df", len(df_test))
    else:
        df_test = None
        metrics = ['model_val_elbo', 'train_rmse', 'es_val_rmse', 'model_val_rmse', 'train_r2', 'es_val_r2',
                   'model_val_r2',
                   'model_val_negloglik', 'kl_elbo', 'mix_res_stddev',
                   'mix_res_mean', 'epis_res_stddev', 'epis_res_mean', 'aleo_res_stddev', 'aleo_res_mean', 'caos',
                   'caus',
                   'stopped_epoch', 'best_epoch', 'nn_time', 'bnn_time', 'pred_time']
    features_len = len(df_train.columns) - len(target)
    return df_train, df_test, metrics, features_len, target


def loss_plot_with_nn(hist_bnn, hist_nn, filtering=5):
    """
    Concatenates NN pretraining loss plot with BNN loss plot.
    :param hist_bnn: pd.DataFrame. The loss history of the BNN
    :param hist_nn: The loss history of the NN
    :param filtering: The degree of smoothing of the loss curves. Must be odd and larger than 3.
    """

    if filtering <= 3:
        filtering = 5
    fig, ax = plt.subplots(figsize=(17, 9), constrained_layout=True)

    argmin_epoch_nn = np.argmin(hist_nn['val_loss'])
    argmin_epoch_bnn = np.argmin(hist_bnn['val_loss'])
    min_epoch_nn = hist_nn['epoch'][argmin_epoch_nn]
    min_epoch_bnn = hist_bnn['epoch'].iloc[argmin_epoch_bnn] + min_epoch_nn
    hist_nn_cut = hist_nn[hist_nn['epoch'] <= min_epoch_nn]
    epochs_total = np.append(hist_nn_cut['epoch'], hist_bnn['epoch'] + min_epoch_nn)
    epochs_bnn = hist_bnn['epoch'] + min_epoch_nn
    val_negloglik = np.append(hist_nn_cut['val_loss'].to_numpy(), hist_bnn['val_negloglik'].to_numpy())
    negloglik = np.append(hist_nn_cut['loss'].to_numpy(), hist_bnn['negloglik'].to_numpy())
    kl_elbo = hist_bnn['kl_elbo']
    elbo = hist_bnn['elbo']
    val_elbo = hist_bnn['val_elbo']
    metrics = [negloglik, val_negloglik, kl_elbo, elbo, val_elbo]
    metrics_label = ['Likelihood Train Loss', r'Likelihood $\mathrm{Validation}_{\mathrm{es}}$ Loss',
                     'Complexity Loss (ELBO-Metric)', 'ELBO-Metric Train',
                     r'ELBO-Metric $\mathrm{Validation}_{\mathrm{es}}$']

    for i, m in enumerate(metrics):
        # metrics[i] = savgol_filter(m, filtering, 3)
        metrics[i] = m
    ax.plot(epochs_total, metrics[0], color=colors['Verdigris'], label=metrics_label[0])
    ax.plot(epochs_total, metrics[1], color=colors['Emerald'], label=metrics_label[1])
    ax.plot(epochs_bnn, metrics[2], color=colors['Fire Opal'], label=metrics_label[2])
    ax.plot(epochs_bnn, metrics[3], color=colors['Maximum Yellow Red'], label=metrics_label[3])
    ax.plot(epochs_bnn, metrics[4], color=colors['Prussian Blue'], label=metrics_label[4])
    ax.plot(epochs_bnn, metrics[4], '*', markevery=[argmin_epoch_bnn], color=colors['Fire Opal'],
            label='Selected Model', markersize=12)
    ax.set_ylim(bottom=-6, top=1)
    ax.set_xlim(right=np.max(epochs_total))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss/Metric')
    ax.axhline(0, linestyle='--', color='black', alpha=0.2, lw=0.8)
    ax.axvline(min_epoch_nn, linestyle='-', color='black', alpha=0.2, lw=0.8, label='BNN Start')
    ax.legend()
    plt.savefig('Loss_plot_bnn_and_nn.png', dpi=300)


# hist_bnn = pd.read_csv(
#     '/home/per-dimitri/Dropbox/Master/BayesianNeuralNetwork/Models/saved_bnns/testing/model_history_table'
# )
# hist_bnn = hist_bnn.drop(['stopped_epoch'], axis=1)
#
# with open('/home/per-dimitri/Dropbox/Master/BayesianNeuralNetwork/Models/saved_bnns/testing/model_nn_history', "rb") as f:
#     model_nn_history = pickle.load(
#         f)
# hist_nn = pd.DataFrame(model_nn_history)
# loss_plot_with_nn(hist_bnn=hist_bnn, hist_nn=hist_nn)

def activation_comparison(df, upper_limit=200):
    """
    Compares various activation functions and their performance in a scatter plot.
    :param df: The dataframe with performance values. Must include performance values for LeakyRelu, relu, sigmoid, swish, mish, tanh, selu, elu
    :param upper_limit: Upper limit of the validation ELBO score. Everything above will not be shown.
    """
    plt.rc('xtick', labelsize=22)  # fontsize of the x tick labels
    fig, ax = plt.subplots(1, 3, figsize=(17, 9), sharex=True)
    ax = ax.flatten()
    df = df.sort_values(by=['model_val_elbo(scaled)*'])
    df = df[df['model_val_elbo(scaled)*'] < upper_limit]
    df.loc[(df.activation == 'LeakyRelu'), 'activation'] = 'LRelu'
    df.loc[(df.activation == 'relu'), 'activation'] = 'ReLU'
    df.loc[(df.activation == 'sigmoid'), 'activation'] = 'Sig.'
    df.loc[(df.activation == 'swish'), 'activation'] = 'Swish'
    df.loc[(df.activation == 'mish'), 'activation'] = 'Mish'
    df.loc[(df.activation == 'tanh'), 'activation'] = 'Tanh'
    df.rename(columns={'activation': 'Activation Function', 'model_val_elbo(scaled)*': 'ELBO-Validation',
                       'model_val_rmse': 'RMSE-Validation'}, inplace=True)
    df['CAOS-CAUS'] = df['caos'] - df['caus']
    sns.set_palette(palette=list(colors.values()))
    ax0 = sns.stripplot(x='Activation Function', y='ELBO-Validation', data=df, ax=ax[0], linewidth=1, size=9)
    ax1 = sns.stripplot(x='Activation Function', y='RMSE-Validation', data=df, ax=ax[1], linewidth=1, size=9)
    ax2 = sns.stripplot(x='Activation Function', y='CAOS-CAUS', data=df, ax=ax[2], linewidth=1, size=9)
    fig.tight_layout()
    ax[0].grid(axis='both')
    ax[1].grid(axis='both')
    ax[2].grid(axis='both')
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=35)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=35)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35)
    ax0.set_xlabel('')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    plt.savefig('Activation-Comparison.png' % target, dpi=300)


def coverage_plot(model, n=100, interval_type='ETI', savename=None, ax=None, large_ix=None, rel_log_errors=False):
    """
    Plots the percentage of test values are within a certain ETI for n different ETIs between 0 and 100%
    :param model: obj.
    The Initialization.py object.
    :param n: int.
    How many ETIs between 0 and 100% to run over. Note that an integration is performed over the curve, so to get correct CAOS/CAUS score a sufficient number is needed.
    :param interval_type: string.
    Type of interval, ETI or HDI. Note that HDI takes very long time compared to ETI, so too high n is not recommended if you're short on time.
    :param savename: string.
    Name of png image to save. None to not save.
    :param ax: matplotlib subplot object.
    Pass in ax if plotted together with multiple subplots outside method.
    :param large_ix: list.
    list of indicies to remove data.
    :return: ax
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 9), constrained_layout=True)
    _, y_pred_samples, _, _ = model.get_predictions()
    y_test, _, _ = model.get_y_test_and_mix()

    if large_ix is not None:
        for m, i in enumerate(large_ix):
            y_pred_samples = np.delete(y_pred_samples, i, axis=1)
            y_test = y_test.drop(y_test.index[i])
            for f, k in enumerate(large_ix):
                if k > i:
                    large_ix[f] = k - 1

    coverage_deviation_score, CAOS, CAUS, percentiles, pred_percentiles = sm.area_deviation_from_ideal_coverage(
        y_pred_samples, y_test,
        interval_type=interval_type,
        resolution=n, get_percentiles=True, rel_log_errors=rel_log_errors)
    ax.plot(percentiles, pred_percentiles,
            label=interval_type + ' Curve', color=colors['Fire Opal'])
    ax.plot(percentiles, percentiles, label='Ideal Curve', color=colors['Emerald'])
    ax.fill_between(percentiles, percentiles, pred_percentiles,
                    where=pred_percentiles >= percentiles, color=colors['Verdigris'], alpha=0.2,
                    label='CAOS: %s' % (round(CAOS, 5)))
    ax.fill_between(percentiles, pred_percentiles, percentiles,
                    where=pred_percentiles < percentiles, color=colors['Maximum Yellow Red'], alpha=0.2,
                    label='CAUS: %s' % (round(CAUS, 5)))
    ax.set_xlabel('Percentiles')
    ax.set_ylabel('Percentile Coverage')
    ax.legend()
    # ax.tick_params(axis='x', which='minor', bottom=False)
    minor_ticks = np.arange(5, 100, 5)
    minor_ticks_labels = [5, 10, 15, 25, 30, 35, 45, 50, 55, 65, 70, 75, 85, 90, 95]
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(b=True, which='major', color='grey', linestyle='-')
    ax.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.2)
    ax.set_xticklabels(minor_ticks_labels, minor=True, fontsize=12)
    ax.set_yticklabels(minor_ticks_labels, minor=True, fontsize=12)
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=0, right=100)
    if savename is not None:
        plt.savefig(savename, dpi=300)
    return ax


def filter_by_cv(remove_points, keep_under, err, y_pred, y_test, y_pred_samples, mix_df_test=None, X_test=None,
                 log_bool=True):
    j = 0
    if isinstance(remove_points, int) and remove_points != 0:
        CV = np.zeros((2, len(err[0])))
        for i in range(len(err[0])):
            if log_bool:
                err_l = np.abs(err[0][i] / np.median(np.log10(y_pred)))
                err_u = np.abs(err[1][i] / np.median(np.log10(y_pred)))
                if i == 0:
                    print("Median: ", np.median(np.log10(y_pred)))
            else:
                err_l = np.abs(err[0][i] / np.median(y_pred))
                err_u = np.abs(err[1][i] / np.median(y_pred))
                if i == 0:
                    print("Median: ", np.median(y_pred))
            CV[0][i] = err_l
            CV[1][i] = err_u
        large_ix = np.argpartition(-(CV[0] + CV[1]), range(remove_points))[:remove_points]
        stored_large_ix = large_ix.copy()
        report_large_ix = []
        for m, i in enumerate(large_ix):
            if keep_under is not None:
                if CV[0][i] * 100 > keep_under or CV[1][i] * 100 > keep_under:
                    report_large_ix.append(stored_large_ix[m])
                    print("Removed CV: %s%%, %s%%" % (CV[0, i] * 100, CV[1, i] * 100))
                    print("Removed STD: %s" % err[:, i])
                    print("True value: ", y_test.iloc[i].to_numpy().ravel())
                    print("Prediction value: ", y_pred[i])
                    print("\n")
                    CV = np.delete(CV, i, axis=1)
                    err = np.delete(err, i, axis=1)
                    y_pred_samples = np.delete(y_pred_samples, i, axis=1)
                    y_pred = np.delete(y_pred, i)
                    if mix_df_test is not None:
                        mix_df_test = mix_df_test.drop(mix_df_test.index[i])
                    if X_test is not None:
                        X_test = X_test.drop(X_test.index[i])
                    y_test = y_test.drop(y_test.index[i])
                    j += 1
                    for f, k in enumerate(large_ix):
                        if k > i:
                            large_ix[f] = k - 1

            else:
                print("Removed relative error (in percentage): %s %%" % errors[:, i])
                print("True value: ", y_test.iloc[i].to_numpy().ravel())
                print("Prediction value: ", y_pred[i])
                print("\n")
                err = np.delete(err, i, axis=1)
                y_pred_samples = np.delete(y_pred_samples, i, axis=1)
                y_pred = np.delete(y_pred, i)
                if mix_df_test is not None:
                    mix_df_test = mix_df_test.drop(mix_df_test.index[i])
                if X_test is not None:
                    X_test = X_test.drop(X_test.index[i])
                y_test = y_test.drop(y_test.index[i])
                j += 1
        print("\nNumber of removed errors: ", j)
        print("\n")
    return err, y_pred, y_test, y_pred_samples, mix_df_test, X_test, report_large_ix, j


def true_vs_pred(model, error_type='eti', perc=95, spaghetti_num=5, filt_fb=None, log_bool=True, rel_log_errors=False,
                 ax=None, remove_points=None, keep_under=None, mixing_plot=True, train=False, std_multiplier=1,
                 use_band=False, prune=None, inset_bool=False):
    """
    Plots the true val/test value against their predictions
    :param use_band:
    :param model: Initialization.py obj.
    The loaded trained model.
    :param error_type: string.
    The error type used. Choosen between 'eti', 'mix', 'aleo', 'epis'
    :param perc: int.
    The equal tailed interval (eti) coverage percetange. No effect if 'eti' not chosen.
    :param spaghetti_num: int.
     Number of samples shown in the true vs pred plot. Gives an impression of the distribution around the final predictions.
    :param filt_fb: float.
    filters out all cross sections over this value. fb for femtobarn.
    :param log_bool: boolean.
    Whether to log predictions and errors.
    :param rel_log_errors:
    Whether to use first order approximation on errors. Use with error_type='eti'.
    :param ax: matplotlib subplot object.
    Pass in ax if plotted together with multiple subplots outside method.
    :param remove_points: int.
    Number of points to remove, given the condition "keep_under"
    :param keep_under:
    Removes data points if over keep_under = (point error)/(global median), given the number of data points in remove_points.
    :param mixing_plot: boolean.
    :param train: boolean.
    :param std_multiplier: int.
    Whether to include colouring according to particle type.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 9))
        axs = [ax]
    else:
        axs = [ax]

    # Returns unlogged:
    if train:
        _, _, mix_df_train = model.get_y_test_and_mix()
        _, y_test, _, _ = model.get_data()
        _, _, y_pred, y_pred_samples = model.get_predictions()
        mix_df_test = mix_df_train
        y_sigma_mix, y_sigma_epis, y_sigma_aleo = model.get_uncertainties(inv_errors=not log_bool, log10=True,
                                                                          train=True)
    else:
        y_test, mix_df_test, mix_df_train = model.get_y_test_and_mix()
        y_pred, y_pred_samples, _, _ = model.get_predictions()
        y_sigma_mix, y_sigma_epis, y_sigma_aleo = model.get_uncertainties(inv_errors=not log_bool, log10=True,
                                                                          train=False)
    data_len = len(y_test)
    if error_type == 'epis':
        err = std_multiplier * y_sigma_epis
        title = r'$\mathrm{Epistemic} \ \sigma_{\mathrm{e}}$'
    elif error_type == 'aleo':
        err = std_multiplier * y_sigma_aleo
        title = r'$\mathrm{Aleoteric} \ \sigma_{\mathrm{a}}$'
    elif error_type == 'mix':
        err = std_multiplier * y_sigma_mix
        title = r'$\mathrm{Mixture} \ \sigma_{\mathrm{m}}$'
    else:
        eti = sm.eti(y_pred_samples, perc=perc)
        err = eti
        title = 'ETI %s%%' % perc

    # if rel_log_errors:
    #     err = np.log10(np.exp(1)) * (err / y_pred)
    #
    if remove_points is not None and keep_under is not None:
        err, y_pred, y_test, y_pred_samples, mix_df_test, _, large_ix, j = filter_by_cv(remove_points, keep_under, err,
                                                                                        y_pred,
                                                                                        y_test, y_pred_samples, X_test=None, mix_df_test=mix_df_test)
    else:
        large_ix, j = None, 0

    r2 = R2(y_test, y_pred)
    rmse = MSE(y_test, y_pred, squared=False)
    if not mixing_plot:
        mix_df_test = None
    ax, error_mean, largest_df = pp.true_predict_plot(y_test, y_pred, err, ax=axs[0],
                                                      y_pred_samples=y_pred_samples[:spaghetti_num], mix_df=mix_df_test,
                                                      label_error=title, filt_fb=filt_fb, log_bool_data=log_bool,
                                                      use_band=use_band, prune=prune, inset_bool=inset_bool)

    textstr = 'Removed Points: %s\n Percentage Removed: %s' % (j, round(j / data_len * 100,
                                                                        4)) + '%\n' + 'Unscaled scores:\n' r'$\mathrm{R}^2=%.6f$' '\n' r'$\mathrm{RMSE}=%.6f$' % (
              r2, rmse)
    props = dict(boxstyle='round', facecolor='white', alpha=0.2)
    # place a text box in upper left in axes coords
    ax.text(+0.7, +0.0215, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.2)
    print("Largest: \n", largest_df)
    print("Error mean: ", error_mean)
    return ax, large_ix


def pred_vs_mass(model, mass='m1000022', error_type='eti', perc=95, remove_points=None, keep_under=None, ax=None,
                 log_bool=True, std_multiplier=1):
    """
    :param model:
    :param mass:
    :param error_type:
    :param perc:
    :param remove_points:
    :param keep_under:
    :param ax:
    :param log_bool:
    :param std_multiplier:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 9))

    y_pred, y_pred_samples, _, _ = model.get_predictions()
    X_train, y_train, X_test, y_test = model.get_data()
    y_sigma_mix, y_sigma_epis, y_sigma_aleo = model.get_uncertainties(inv_errors=not log_bool, log10=True)


    if error_type == 'epis':
        err = std_multiplier * y_sigma_epis
        title = r'$\sigma_{\mathrm{Epistemic}}$'
    elif error_type == 'aleo':
        err = std_multiplier * y_sigma_aleo
        title = r'$\sigma_{\mathrm{Aleoteric}}$'
    elif error_type == 'mix':
        err = std_multiplier * y_sigma_mix
        title = r'$\sigma_{\mathrm{mixture}}$'
    else:
        eti = sm.eti(y_pred_samples, perc=perc)
        err = eti
        title = 'ETI %s' % perc
    if remove_points is not None and keep_under is not None:
        _, y_pred, y_test, y_pred_samples, _, X_test, large_ix, _ = filter_by_cv(remove_points, keep_under, y_sigma_mix,
                                                                                 y_pred, y_test, y_pred_samples,
                                                                                 X_test=X_test)
        err = np.delete(err, [large_ix], axis=1)

    sigmas = [err]
    sigmas_sorted = []

    df_test = pd.DataFrame({'X_test': np.abs(X_test[mass]), 'y_test': y_test[y_test.columns[0]],
                            'y_pred': y_pred}, columns=['X_test', 'y_test', 'y_pred']).sort_values(by='X_test')
    for s in sigmas:
        sigma_ = pd.DataFrame({'1': np.array(s[0, :]), '2': np.array(s[1, :])}, index=y_test.index, columns=['1', '2'])
        sigma = sigma_.loc[df_test.index].values.transpose()

        sigmas_sorted.append(sigma)
    err = sigmas_sorted[0]

    errors = {title: err}
    if error_type == 'eti':
        error_log = True
    else:
        error_log = False
    pp.fill_between_plot(ax, df_test, error_dict=errors, log_bool=log_bool, error_log=error_log)
    return large_ix


def rel_error(model, large_ix=None):
    y_test, _, _ = model.get_y_test_and_mix()
    y_pred, y_pred_samples, _, _ = model.get_predictions()
    y_sigma_mix, y_sigma_epis, y_sigma_aleo = model.get_uncertainties(inv_errors=True, log10=False, train=False)
    if large_ix is not None:
        for m, i in enumerate(large_ix):
            y_pred_samples = np.delete(y_pred_samples, i, axis=1)
            y_test = y_test.drop(y_test.index[i])
            y_sigma_mix = np.delete(y_sigma_mix, i, axis=1)
            for f, k in enumerate(large_ix):
                if k > i:
                    large_ix[f] = k - 1
    rel_error = np.mean(np.abs(y_sigma_mix[0, :].ravel() + y_sigma_mix[1, :].ravel()) / y_test.to_numpy().ravel())
    print(rel_error)


def time_vs_metric(model, path_model, large_ix, step_size):
    model.load_model(path_model, sample_size=1, load_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    ax = axes.flatten()
    sample_sizes = np.arange(10, 5000, step_size)
    times = np.zeros(len(sample_sizes))
    rmses = np.zeros((len(sample_sizes)))
    rel_errors = np.zeros((len(sample_sizes)))
    for i, s in enumerate(sample_sizes):
        print("sample:", s)
        y_pred, y_pred_samples, y_test, mix, epis, aleo, pred_time = model.predict_standalone(sample_size=s,
                                                                                              X_test=model.X_test,
                                                                                              invert_log=True,
                                                                                              invert_errors=True)
        times[i] = pred_time
        if large_ix is not None:
            for j in large_ix:
                y_pred = np.delete(y_pred, j, axis=0)
                y_test = y_test.drop(y_test.index[j])
                mix = np.delete(mix, [j], axis=1)
                for f, k in enumerate(large_ix):
                    if k > i:
                        large_ix[f] = k - 1
        rmse = MSE(y_pred, y_test, squared=False)
        rel_errors[i] = np.mean(np.abs(mix[0, :].ravel() + mix[1, :].ravel()) / y_test.to_numpy().ravel())
        rmses[i] = rmse

    x = ax[0].scatter(sample_sizes, rmses, c=times, cmap='viridis', s=22)
    ax[1].scatter(sample_sizes, rel_errors, c=times, cmap='viridis', s=22)
    ax[0].set_xlabel('Sample Size')
    ax[1].set_xlabel('Sample Size')
    ax[0].set_ylabel('RMSE')
    ax[1].set_ylabel('ARTE')
    fig.subplots_adjust(wspace=0.3)
    cbar = fig.colorbar(x, ax=axes.ravel().tolist())
    cbar.set_label('Time (sec)')
    ax[0].grid(alpha=0.5)
    ax[1].grid(alpha=0.5)
    fig.tight_layout()
    plt.savefig('RMSERtrue', dpi=300)
    print(len(y_test))


def standardised_residuals(model, save_name, eti_bool=False, perc=95, ax=None):

    y_test, _, _ = model.get_y_test_and_mix()
    y_pred, y_pred_samples, _, _ = model.get_predictions()
    y_sigma_mix, y_sigma_epis, y_sigma_aleo = model.get_uncertainties(inv_errors=True, log10=False, train=False)

    sigmas = [y_sigma_mix, y_sigma_epis, y_sigma_aleo]
    sigmas_label = [r'$\Delta^{\pm}_{\mathrm{mixture}}$', r'$\Delta^{\pm}_{\mathrm{Epistemic}}$', r'$\Delta^{\pm}_{\mathrm{Aleoteric}}$']
    eti = sm.eti(y_pred_samples, perc=perc)

    cols = list(colors.keys())
    if not eti_bool:
        fig, axs = plt.subplots(1, 3, figsize=(17, 9), constrained_layout=True, sharex=True, sharey=True)
        axs = axs.flatten()
        for i, s in enumerate(sigmas):
            ax = axs[i]
            stand_res, std, mean = sm.standardized_residuals(y_test.to_numpy().ravel(), y_pred.ravel(), s)
            for k, r in enumerate(stand_res):
                if r > 5:
                    stand_res[k] = 5
                elif r < -5:
                    stand_res[k] = -5
            # stand_res = stand_res[np.abs(stand_res) < 1]
            bins = 1000


            ax = sns.histplot(stand_res, bins=bins, stat='density', kde=False,
                               line_kws={'linewidth': 1}, ax=ax, color=colors[cols[i]], alpha=0.5,
                              label=sigmas_label[i] + ', ' + r'$\mu_z=$ %s, $\sigma_z=$ %s' % (round(mean,3),round(std,3)))


            ax.set_xlim(-5.02, 5.02)

            b = np.arange(np.min(stand_res), np.max(stand_res) + 0.1, 0.01)
            # print(len(b))
            standard_norm = stats.norm.pdf(b, 0, 1)
            ax.plot(b, standard_norm, '--', label="Standard Normal",
                    color=colors['Maximum Yellow Red'], lw=2)
            ax.legend()
            ax.set_xlabel('z')
            ax.set_ylabel('Probability Density')
            plt.savefig(save_name, dpi=300)
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(17, 9), constrained_layout=True, sharex=True, sharey=True)
        stand_res, std, mean = sm.standardized_residuals(y_test.to_numpy().ravel(), y_pred.ravel(), eti)
        bins = 500
        stand_res_ = stand_res.copy().ravel()
        for k, r in enumerate(stand_res):
            if r > 1:
                stand_res_ = np.delete(stand_res_, k, axis=0)
            elif r < -1:
                stand_res_ = np.delete(stand_res_, k, axis=0)
        stand_res = stand_res_
        print(np.max(stand_res))
        print(np.min(stand_res))
        print(stand_res)
        stand_res = stand_res[np.abs(stand_res) < 5]
        mean = np.mean(stand_res)
        std = np.std(stand_res)
        ax = sns.histplot(stand_res, bins=bins, stat='density', kde=False,
                          line_kws={'linewidth': 1}, ax=ax, color=colors[cols[0]],
                          alpha=0.5, label=r'$\Delta^{\pm}_{\mathrm{ETI 95}}$' + ', ' + r'$\mu_z=$ %s, $\sigma_z=$ %s' % (round(mean,3),round(std,3)))

        ax.set_xlim(-5, 5)

        b = np.arange(np.min(stand_res), np.max(stand_res) + 0.1, 0.01)
        standard_norm = stats.norm.pdf(b, 0, 1)
        ax.plot(b, standard_norm, '--', label="Standard Normal",
                color=colors['Maximum Yellow Red'], lw=2)

        ax.legend()
        ax.set_xlabel('z')
        ax.set_ylabel('Probability Density')
        plt.savefig(save_name, dpi=300)



#########################################################################################################

def plotting(model, plot='coverage', savename=None):
    if plot == 'coverage':
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(19, 11), constrained_layout=True)
        axs = axs.flatten()
        _, large_ix = true_vs_pred(model, filt_fb=5000, error_type='epis', perc=95, remove_points=6, keep_under=1,
                                   log_bool=True, rel_log_errors=True, ax=axs[0], use_band=True, mixing_plot=True, std_multiplier=3)
        coverage_plot(model, n=300, savename=None, interval_type='ETI', ax=axs[1], large_ix=large_ix)
        axs[0].set_ylim(-6, 4)
        if savename is not None:
            plt.savefig(savename, dpi=300)
        return large_ix
    elif plot == 'pred_remove':
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(19, 11), sharey=True, sharex=True, constrained_layout=True)
        axs = axs.flatten()
        true_vs_pred(model, filt_fb=5000, error_type='mix', perc=95, remove_points=1, keep_under=1, log_bool=True,
                     ax=axs[0], use_band=True, prune=0.5)
        true_vs_pred(model, filt_fb=5000, error_type='mix', perc=95, remove_points=20, keep_under=1, log_bool=True,
                     ax=axs[1], use_band=True, prune=0.5)
        axs[1].set_ylabel('')
        if savename is not None:
            plt.savefig(savename, dpi=300)
    elif plot == 'test':
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(19, 11), constrained_layout=True)
        _, large_ix = true_vs_pred(model, filt_fb=5000, error_type='mix', perc=95, remove_points=1000, keep_under=7,
                                   log_bool=True, ax=axs, use_band=True, inset_bool=True)
        if savename is not None:
            plt.savefig(savename, dpi=300)
    elif plot == 'test_2324':
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(19, 11), constrained_layout=True)
        _, large_ix = true_vs_pred(model, filt_fb=5000, error_type='mix', perc=95, remove_points=1000, keep_under=7,
                                   log_bool=True, ax=axs, mixing_plot=False)
        plt.savefig('PredvsTrue2324Test', dpi=300)
        return large_ix
    elif plot == 'train':
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(19, 11), constrained_layout=True)
        true_vs_pred(model, filt_fb=5000, error_type='mix', perc=95, remove_points=1000, keep_under=7, log_bool=True,
                     ax=axs, train=True)
        plt.savefig('PredvsTrue2222TrainofTest', dpi=300)
    elif plot == 'test_mass':
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(19, 11), constrained_layout=True)
        pred_vs_mass(model, error_type='mix', ax=axs, remove_points=20, keep_under=7, log_bool=True)
        plt.savefig('PredVsMass2222Test', dpi=300)
    elif plot == 'test_mass_2324':
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(19, 11), constrained_layout=True)
        large_ix = pred_vs_mass(model, error_type='eti', ax=axs, remove_points=1, keep_under=200, log_bool=True,
                                mass='m1000023', std_multiplier=2)
        axs.set_xlabel(r'$m_{\widetilde{\chi}_{2}^{0}}$')
        axs.set_ylabel(r'$\log_{10}\left(\hat{\sigma} / \sigma_{0}\right), \sigma_{0}=1 \mathrm{fb}$')
        plt.legend()
        plt.savefig('PredVsMass2324Test', dpi=300)
        return large_ix


############################################################################
# Set xsec
xsec = '2222'
# Set if testing or validation
testing = False
# Set if filter vmix11 and nmix21
filter_bool = False

df_train, df_test, metrics, features_len, target = set_data(testing=testing, xsec=xsec)

if xsec == '2324':
    mix_bool=False
    if filter_bool:
        df_test = df_test[np.abs(df_test['vmix11']) < 0.75]
        df_test = df_test[np.abs(df_test['nmix21']) < 0.75]
else:
    mix_bool=True


############## Declaring priors and posterior mean field distributions ##############
nn_prior = dist_init.SetNNDistribution(dtype=dtype, trainable=False, dist='Gaussian')
nn_prior.set_nn_constant_initializer()

nn_pmf = dist_init.SetNNDistribution(dtype, trainable=True, link_object=nn_prior, dist='Gaussian')
nn_pmf.set_nn_complex_initializer()

model = init.Initialization(df_train, data_test=df_test, target_name=target, prior=nn_prior, pmf=nn_pmf,
                            test_fracs=[0.1875, 0.1875],
                            seed=seed, log_scaling=True, affine_scaler='MinMaxScaler', affine_target_scaling=True,
                            monitored_metrics=metrics,
                            model_name='model_visualization', mean_bool=True,
                            sort_permutations_by='layers', mix_fix=mix_bool)

""" Available models. """
current_dir = os.path.dirname(os.path.abspath(__file__))

# path_model = '/empirical/saved_bnns/model_training_0'
path_model = '/eba_false_small_kappa/saved_bnns/model_training_7'
# path_model = '/2222_test/testing'
# path_model = '/2324_test/saved_bnns/testing'
model.load_model(current_dir + path_model, sample_size=5000)


""" Example plots: Change xsec or testing/validation above. """

plotting(plot='coverage', model=model, savename='PredvsTrueCAOS2222_meantrue_epis')

# plotting(plot='test_2324', model=model)
# plotting(plot='coverage', model=model, savename='PredvsTrueCAOS2222_meantrue')
# plotting(plot='pred_remove', model=model, savename='PredvsTrue2222Val')
# plotting(plot='test_mass_2324', model=model, savename='PredVsMass2324Test')
# plotting(plot='test', model=model, savename='PredvsTrue2222Test')

pred_time = time.time() - t
plt.show()
