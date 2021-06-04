import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
import numpy as np
import seaborn as sns
import sys
import scipy.stats as stats
sys.path.insert(1, '/home/per-dimitri/Dropbox/Master/BayesianNeuralNetwork/Models')
# noinspection PyUnresolvedReferences
from bnn_module import data_handling as dh

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
plt.rc('legend', fontsize=18) #fontsize of the legend


colors = {'Fire Opal': '#EE6352', 'Emerald': '#59CD90', 'Cerulean Crayola': '#3FA7D6', 'Maximum Yellow Red': '#FAC05E', 'Vivid Tangerine': '#F79D84', 'Prussian Blue': '#173753'}

dtype = "float64"
# dataset_name = 'EWonly_PMCSX_22-22_filt'
# dataset_name = 'EWonly_PMCSX_22-22_23-37_35-24_25-24_filt'
dataset_name = 'EWonly_PMCSX_23-24_23--24_22-22_23-37_35-24_25-24_25--24_22--37_filt'
# dataset_name = 'EWonly_PMCSX_22-22'
target = ["1000022_1000022_13000_NLO_1"]

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dataset_path = dir_path + '/Data Harvest/' + dataset_name


df = pd.read_csv(dataset_path, sep="\t", skipinitialspace=True, index_col=0)

# df = df.drop(columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
# df_test = df_test.drop(columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
features_len = len(df.columns) - len(target)
df4 = df[['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', target[0]]]

def parameter_scatter_2d(target, target2=None, label1=None, label2=None):
    fig, ax = plt.subplots(2, 2, figsize=(17, 9))
    axs = ax.flatten()

    c = ['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)']
    plt_c = [r'$\tan(b)$', r'$M_1$', r'$M_2$', r'$\mu$']

    for i, p in enumerate(c):
        if i == 0 and label1 is not None:
            lab1 = label1
        x = axs[i].scatter(df[p], np.log10(df[target]), s=2, color=colors['Fire Opal'], alpha=0.4, label=lab1)
        if target2 is not None:
            if i == 0 and label2 is not None:
                lab2 = label2
            x = axs[i].scatter(df[p], np.log10(df[target2]), s=2, color=colors['Emerald'], alpha=0.4, label=lab2)
        axs[i].set_xlabel(r"$%s$" % plt_c[i])
    axs[0].set_ylabel(r"$\log _{10}\left(\sigma / \sigma_{0}\right), \sigma_{0}=1 \mathrm{fb}$")
    axs[2].set_ylabel(r"$\log _{10}\left(\sigma / \sigma_{0}\right), \sigma_{0}=1 \mathrm{fb}$")

    fig.subplots_adjust(wspace=0.22, hspace=0.35)
    fig.suptitle(r'')
    legend = axs[0].legend(bbox_to_anchor=(1.65, 1.25), ncol=2, markerscale=5, fontsize=20)
    plt.setp(legend.get_texts(), fontsize='20', va='center')
    plt.savefig('parameter_2scatterings_%s.png' % target, dpi=300)

def parameter_scatter(target, df, filter=1000, filter_col='mu(MX)'):

    fig, ax = plt.subplots(3, 2, figsize=(17, 9))
    axs = ax.flatten()

    c = ['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)']
    plt_c = [r'$\tan(b)$', r'$M_1$', r'$M_2$', r'$\mu$']

    df = df.loc[(filter >= np.abs(df[filter_col]))]
    df.dropna()

    pairs = list(itertools.combinations(c, 2))
    plt_pairs = list(itertools.combinations(plt_c, 2))
    for i, p in enumerate(pairs):
        x = axs[i].scatter(df[p[0]], df[p[1]], c=np.log10(np.abs(df[target])), cmap='viridis', s=12)
        axs[i].set_xlabel(r"$%s$" % plt_pairs[i][0])
        axs[i].set_ylabel(r"$%s$" % plt_pairs[i][1])

    fig.subplots_adjust(wspace=0.19, hspace=0.35)
    cbar = fig.colorbar(x, ax=axs.tolist())
    cbar.set_label(r'$m_{\chi_1^0}$')
    plt.savefig('parameter_scatter.png', dpi=300)


# parameter_scatter('1000022_1000022_13000_NLO_1', filter=200, df=df)


def parameter_hist():
    fig, ax = plt.subplots(2, 2, figsize=(17, 9))
    axs = ax.flatten()

    c = ['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)']
    plt_c = [r'$\tan(b)$', r'$M_1$', r'$M_2$', r'$\mu$']
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=100, ec="k", color=colors['Fire Opal'])

    for i, p in enumerate(c):
        x = axs[i].hist(df[p], **kwargs)
        axs[i].set_xlabel(r"$%s$" % plt_c[i])
    axs[0].set_ylabel("Probability Density")
    axs[2].set_ylabel("Probability Density")
    fig.subplots_adjust(wspace=0.25, hspace=0.35)
    plt.savefig('parameter_histogram.png', dpi=300)

# parameter_hist()


def violin_plot_masses(df):
    sns.set(style="ticks", font_scale=2)
    sns.set_style({'font.family': 'serif', 'font.serif': 'Computer Modern Roman', 'text.usetex': True})
    fig, ax = plt.subplots(figsize=(17, 9))
    cols = ['m1000022', 'm1000023', 'm1000025', 'm1000035', 'm1000024', 'm1000037']
    df = np.abs(df[cols])
    ticks_names = [r'$\chi^0_1$', r'$\chi^0_2$', r'$\chi^0_3$', r'$\chi^0_4$', r'$\chi^{\pm}_1$', r'$\chi^{\pm}_2$']
    df = df.melt(var_name='Particle', value_name='Mass (GeV)')
    print(df)
    sns.set_palette(palette=list(colors.values()))
    sns.violinplot(ax=ax, x=r'Particle', y=r'Mass (GeV)', data=df)
    for violin, alpha in zip(ax.collections[::2], [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]):
        violin.set_alpha(alpha)
    ax.set_xticklabels(ticks_names, size=24)
    plt.grid()
    plt.savefig('parameter_violin.png', dpi=300)

# violin_plot_masses(df)


dataset_train_name = 'EWonly_PMCSX_22-22_train'
dataset_test_name = 'EWonly_PMCSX_22-22_test'

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dataset_train_path = dir_path + '/Data Harvest/' + dataset_train_name
dataset_test_path = dir_path + '/Data Harvest/' + dataset_test_name

df_train = pd.read_csv(dataset_train_path, sep="\t", skipinitialspace=True, index_col=0)
df_test = pd.read_csv(dataset_test_path, sep="\t", skipinitialspace=True, index_col=0)
df_train = df_train.drop(columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])
df_test = df_test.drop(columns=['tanb', 'M_1(MX)', 'M_2(MX)', 'mu(MX)', 'nmix21', 'nmix22', 'nmix23', 'nmix24'])

def plot_scales(df_train, df_test, target):
    labels = [r"$\left(\sigma / \sigma_{0}\right), \sigma_{0}=1 \mathrm{fb}$",
              r"$\ln\left(\sigma / \sigma_{0}\right), \sigma_{0}=1 \mathrm{fb}$",
              r"$\mathrm{M}\left(\ln\left(\sigma / \sigma_{0}\right)\right), \sigma_{0}=1 \mathrm{fb}$"]
    df_test = df_test.sort_values(by=target)
    df_train = df_train.sort_values(by=target)
    X_test, y_test = df_test.drop(
        columns=target), df_test[target]
    X_train, y_train = df_train.drop(
        columns=target), df_train[target]
    y_train_log, y_test_log = pd.DataFrame(np.log(y_train)), pd.DataFrame(np.log(y_test))
    X_train, X_test, y_train_alog, y_test_alog, _, _ = dh.scaling(X_train, X_test, y_train_log, y_test_log, scaling='MinMaxScaler', target_scaling=True)

    fig, axs = plt.subplots(1, 3, figsize=(17, 9), constrained_layout=True)
    axs = axs.flatten()

    train = [pd.DataFrame(y_train)[target], y_train_log[target], y_train_alog[target]]
    test = [pd.DataFrame(y_test)[target], y_test_log[target], y_test_alog[target]]

    for i, ax in enumerate(axs):
        print("MAX", np.max(train[i]))
        print("MAX", np.max(test[i]))
        train[i] = (train[i]).to_numpy()
        test[i] = (test[i]).to_numpy()
        if i == 0:
            bins = 10000
        else:
            bins = 100
        ax1 = sns.histplot(train[i], bins=bins, stat='density', kde=False,
                     line_kws={'linewidth': 1}, ax=ax, color=colors['Fire Opal'], alpha=0.5, label='Train Data')
        ax2 = sns.histplot(test[i], bins=bins, stat='density', kde=False,
                     line_kws={'linewidth': 1}, ax=ax, color=colors['Cerulean Crayola'], alpha=0.5, label='Test Data')

        if i > 0:
            mean = np.mean(train[i])
            std = np.std(train[i])
            b = np.arange(np.min(train[i]), np.max(train[i])+0.1, 0.01)
            standard_norm = stats.norm.pdf(b, mean, std)
            ax.plot(b, standard_norm, '--', label="Normal Fit, Mean: %s, Std: %s" % (round(mean, 2), round(std, 2)), color=colors['Maximum Yellow Red'], lw=1.5)

        if i == 0:
            ax1.set(xlim=(0, 20))
            ax2.set(xlim=(0, 20))

        ax.legend()

        ax.set_xlabel(labels[i])
        ax.set_ylabel("")

    axs[0].set_ylabel("Probability Density")
plot_scales(df_train, df_test, target[0])