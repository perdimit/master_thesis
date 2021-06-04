import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


col = {'Fire Opal': '#EE6352', 'Emerald': '#59CD90', 'Verdigris': '#4CBAB3', 'Cerulean Crayola': '#3FA7D6',
       'Maximum Yellow Red': '#FAC05E',
       'Vivid Tangerine': '#F79D84', 'Deep Taupe': '#876A6C', 'Prussian Blue': '#173753'}


def y_by_mix(y, mix_df):
    y_by_mix = [y.loc[mix_df[mix_df['mixing max'] == 'bino'].index],
                y.loc[mix_df[mix_df['mixing max'] == 'wino'].index],
                y.loc[mix_df[mix_df['mixing max'] == 'higgsino'].index]]
    return y_by_mix


def true_predict_plot(y_test, y_pred, sigma, y_pred_samples=None, ax=None, label_error=None, plot_title=None,
                      std_mul=None, mix_df=None, filt_fb=None, log_bool_data=True, use_band=False, prune=None, inset_bool=False):

    if log_bool_data:
        y_test = np.log10(y_test)
        y_pred = np.log10(y_pred)
        y_pred_samples = np.log10(y_pred_samples)
        np.set_printoptions(threshold=np.inf)

    if y_pred_samples is not None:
        spag_size = y_pred_samples.shape[0]
    else:
        spag_size = None

    if filt_fb is not None:
        indicies = y_test.index
        y_test = y_test[y_test[y_test.columns[0]] < filt_fb]
        if mix_df is not None:
            mix_df = mix_df[mix_df.index.isin(y_test.index)]
        y_pred_ = pd.DataFrame(y_pred, index=indicies)
        y_pred = y_pred_.loc[y_test.index].values.ravel()
        sigma_ = pd.DataFrame({'1': np.array(sigma[0, :]), '2': np.array(sigma[1, :])}, index=indicies,
                              columns=['1', '2'])
        sigma = sigma_.loc[y_test.index].values.transpose()
        if spag_size is not None:
            y_pred_samples_new = np.zeros((spag_size, len(y_test)))
            for i in range(spag_size):
                y_pred_s = pd.DataFrame(y_pred_samples[i, :], indicies)
                y_pred_samples_new[i, :] = y_pred_s.loc[y_test.index].values.ravel()
            y_pred_samples = y_pred_samples_new
    error_mean = 0
    errors = np.zeros(len(sigma[0]))
    for i in range(len(sigma[0])):
        error = sigma[0][i] + sigma[1][i]
        errors[i] = error
        error_mean += error
    error_mean /= len(sigma[0])
    num_errors = 20
    if log_bool_data:
        rel_error = np.abs(y_test.to_numpy().transpose().ravel() - y_pred)
    else:
        rel_error = np.abs(y_test.to_numpy().transpose().ravel() - y_pred) / y_test.to_numpy().transpose().ravel()
    large_ix = np.argpartition(-rel_error, num_errors)[:num_errors]
    largest_df = pd.DataFrame({'y_test': y_test.to_numpy().transpose().ravel()[large_ix], "y_pred": y_pred[large_ix],
                               "rel_error": rel_error[large_ix], "error": errors[large_ix]},
                              columns=['y_test', 'y_pred', 'rel_error', 'error'], index=large_ix)
    largest_df = largest_df.sort_values(by='rel_error', ascending=False)

    if ax is None:
        fig, ax = plt.subplots()

    if inset_bool:
        axins = inset_axes(ax, 4, 4, loc='center left', bbox_to_anchor=(0.1, 0.45, 0.5, 0.5), bbox_transform=ax.figure.transFigure)

    if use_band:
        df_test = pd.DataFrame({'y_test': y_test[y_test.columns[0]],
                                'y_pred': y_pred}, columns=['y_test', 'y_pred']).sort_values(by='y_test')
        sig = pd.DataFrame({'1': np.array(sigma[0, :]), '2': np.array(sigma[1, :])}, index=y_test.index,
                              columns=['1', '2'])
        sigma = sig.loc[df_test.index].values.transpose()
        for i in range(y_pred_samples.shape[0]):
            y_pred_samples[i, :] = pd.DataFrame(y_pred_samples[i, :], index=y_test.index).loc[df_test.index].to_numpy().ravel()

        y_test = pd.DataFrame(df_test['y_test'])
        y_pred = df_test['y_pred'].to_numpy().ravel()

        lower = y_pred - sigma[0]
        upper = y_pred + sigma[1]
        ax.fill_between(y_test.to_numpy().ravel(), upper, lower, alpha=0.1, label=label_error, facecolor=col['Verdigris'])
        ax.plot(y_test.to_numpy().ravel(), lower, alpha=0.4, color=col['Verdigris'])
        ax.plot(y_test.to_numpy().ravel(), upper, alpha=0.4, color=col['Verdigris'])
        if inset_bool:
            axins.fill_between(y_test.to_numpy().ravel(), upper, lower, alpha=0.1, label=label_error,
                               facecolor=col['Verdigris'])
            axins.plot(y_test.to_numpy().ravel(), lower, alpha=0.4, color=col['Verdigris'])
            axins.plot(y_test.to_numpy().ravel(), upper, alpha=0.4, color=col['Verdigris'])
        if prune is not None:
            sigma_bar = sigma.copy()
            y_pred_bar = y_pred.copy()
            y_test_bar = y_test.copy()
            del_indx = []
            for i in range(sigma.shape[1]):
                if np.abs(sigma[0][i] + sigma[1][i]) < prune:
                    del_indx.append(i)
            sigma_bar = np.delete(sigma_bar, del_indx, axis=1)
            y_pred_bar = np.delete(y_pred_bar, del_indx, axis=0)
            y_test_bar = y_test_bar.drop(y_test_bar.index[del_indx])
            print("points not to give bar", len(y_pred)-len(y_pred_bar))
            if len(y_pred)-len(y_pred_bar) > 0:
                ax.errorbar(y_test_bar.values.ravel(), y_pred_bar,
                yerr=sigma_bar, capsize=3.5, capthick=1.5, elinewidth=0.2, fmt='none', label=label_error if i == 0 else '',
                color='black', alpha=0.9)
                if inset_bool:
                    axins.errorbar(y_test_bar.values.ravel(), y_pred_bar,
                yerr=sigma_bar, capsize=3.5, capthick=1.5, elinewidth=0.2, fmt='none', label=label_error if i == 0 else '',
                color='black', alpha=0.9)
    else:
        ax.errorbar(y_test.values.ravel(), y_pred,
        yerr=sigma, capsize=3.5, capthick=1, elinewidth=0.2, fmt='none', label=label_error,
        color='black', alpha=0.8)
    s1 = 8
    s2 = 0.01
    ax.set_ylabel(r"$\mathrm{Prediction:} \ \log_{10}\left(\hat{\sigma} / \sigma_{0}\right), \sigma_{0}=1 \mathrm{fb}$")
    ax.set_xlabel(r"$\mathrm{True:} \ \log_{10}\left(\sigma / \sigma_{0}\right), \sigma_{0}=1 \mathrm{fb}$")
    lims = [np.min(y_test), np.max(y_test)]
    ax.plot(lims, lims, alpha=0.7, color=col['Vivid Tangerine'], label='Ideal line', zorder=1)
    if inset_bool:
        axins.plot(lims, lims, alpha=0.7, color=col['Vivid Tangerine'], label='Ideal line', zorder=1)
    if mix_df is None:
        ax.scatter(y_test, y_pred, color='indigo', alpha=0.9, s=s1)
        axins.scatter(y_test, y_pred, color='indigo', alpha=0.9, s=s1)
        if spag_size is not None:
            for y_preds in y_pred_samples[:spag_size]:
                ax.scatter(y_test, y_preds.ravel(), alpha=0.5, s=s2, color='#2573B5')
                axins.scatter(y_test, y_preds.ravel(), alpha=0.5, s=s2, color='#2573B5')
    else:

        y_both = y_test.copy()
        y_both.columns = ['test']
        y_both['pred'] = y_pred
        y_bm = y_by_mix(y_both, mix_df)
        # colors = ['orangered', 'mediumorchid', 'royalblue']
        colors = ['mediumorchid', col['Fire Opal'], col['Cerulean Crayola']]
        labels = ['Bino', 'Wino', 'Higgsino']
        for i in range(len(y_bm)):
            ax.scatter(y_bm[i]['test'], y_bm[i]['pred'], alpha=1, s=s1, label=labels[i], color=colors[i])
            if inset_bool:
                axins.scatter(y_bm[i]['test'], y_bm[i]['pred'], alpha=1, s=s1, label=labels[i], color=colors[i])
        if spag_size is not None:
            for y_pred_sample in y_pred_samples[:spag_size]:
                y_pred_s = pd.DataFrame(y_pred_sample, y_both.index.ravel(), columns=['pred'])
                y_bm_s = y_by_mix(y_pred_s, mix_df)
                for i in range(len(y_bm_s)):
                    ax.scatter(y_bm[i]['test'], y_bm_s[i]['pred'], alpha=0.5, s=s2, color=colors[i])
                    if inset_bool:
                        axins.scatter(y_bm[i]['test'], y_bm_s[i]['pred'], alpha=0.5, s=s2, color=colors[i])
    ax.legend(loc='lower right')
    plt.grid(False)
    if plot_title is not None:
        ax.set_title(plot_title)

    x1, x2, y1, y2 = -0.95, -0.9, -1, -0.8
    if inset_bool:
        axins.set_xlim(x1, x2)  # apply the x-limits
        axins.set_ylim(y1, y2)  # apply the y-limits
        plt.xticks(visible=False)
        plt.yticks(visible=False)

        mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

    return ax, error_mean, largest_df


def fill_between_plot(ax, df_test, error_dict, X_train=None, y_train=None, log_bool=True, error_log=False):
    X_test, y_test, y_pred = df_test['X_test'].values.ravel(), df_test['y_test'].values.ravel(), df_test[
        'y_pred'].values.ravel()
    i = 0
    for key, err in error_dict.items():
        if log_bool:
            y_test = np.log10(y_test)
            y_pred_ = y_pred
            y_pred = np.log10(y_pred)
            if error_log:
                lower = y_pred - np.log10(np.exp(1)) * err[0] / y_pred_
                upper = y_pred + np.log10(np.exp(1)) * err[1] / y_pred_
            else:
                lower = y_pred - err[0]
                upper = y_pred + err[1]
        else:
            lower = y_pred - err[0]
            upper = y_pred + err[1]
        ax.fill_between(X_test.ravel(), upper, lower, alpha=0.1, label='Relative ' + key + '%%' if i == 0 else '', facecolor=col['Cerulean Crayola'])
        ax.plot(X_test, lower, alpha=0.4, color=col['Cerulean Crayola'])
        ax.plot(X_test, upper, alpha=0.4, color=col['Cerulean Crayola'])
        i += 1
    ax.scatter(X_test, y_test, c=col['Prussian Blue'], label='Test Data', s=6)
    ax.scatter(X_test, y_pred, c=col['Fire Opal'], label='Predictions', s=6)
    if X_train is not None and y_train is not None:
        ax.scatter(X_train, y_train, c="black", s=10, alpha=0.5, label='Training data')
    ax.set_xlabel('Mass')
    ax.set_ylabel('Xsec')
    return ax

# def standardised_residuals():

