from __future__ import division
import numpy as np
import scipy.stats.kde as kde
import pandas as pd


def standardized_residuals(y_true, y_pred, error):
    residuals = y_true - y_pred
    std_residuals = np.zeros(np.shape(residuals))
    for i, res in enumerate(residuals):
        if res < 0:
            std_res = res / error[0][i]
        elif res >= 0:
            std_res = res / error[1][i]
        std_residuals[i] = std_res
    return std_residuals, np.std(std_residuals), np.mean(std_residuals)

def ideal_area(upper, lower):
    return (upper**2 - lower**2)/2

def area_deviation_from_ideal_coverage(y_pred_samples, y_true, interval_type='ETI', min_perc=0, max_perc=100, resolution=100, mix_sigmas=None, get_percentiles=False, rel_log_errors=False):
    n = resolution
    percentiles = np.linspace(min_perc, max_perc, n)
    pred_percentiles = np.zeros(n)
    for i, p in enumerate(percentiles):
        pred_percentiles[i] = coverage(y_pred_samples, y_true, perc=p, interval_type=interval_type, rel_log_errors=rel_log_errors)
    over_intervals, over_x_intervals, under_intervals, under_x_intervals = [], [], [], []
    pred_over, pred_under, x_over, x_under = np.array([]), np.array([]), np.array([]), np.array([])
    for i in range(0, len(pred_percentiles)):
        if pred_percentiles[i] > percentiles[i]:
            if pred_percentiles[i-1] > percentiles[i-1]:
                pred_over, x_over = np.append(pred_over, pred_percentiles[i]), np.append(x_over, percentiles[i])
            else:
                over_intervals.append(pred_over)
                over_x_intervals.append(x_over)
                pred_over, x_over = np.append(np.array([]), pred_percentiles[i]), np.append(np.array([]), percentiles[i])
        if i == len(pred_percentiles)-1:
            over_intervals.append(pred_over)
            over_x_intervals.append(x_over)
        elif pred_percentiles[i] < percentiles[i]:
            if pred_percentiles[i-1] < percentiles[i-1]:
                pred_under, x_under = np.append(pred_under, pred_percentiles[i]), np.append(x_under, percentiles[i])
            else:
                under_intervals.append(pred_under)
                under_x_intervals.append(x_under)
                pred_under, x_under = np.append(np.array([]), pred_percentiles[i]), np.append(np.array([]), percentiles[i])
        if i == len(pred_percentiles)-1:
            under_intervals.append(pred_under)
            under_x_intervals.append(x_under)

    area_pred_over = 0
    area_pred_under = 0

    for pred, ideal in zip(over_intervals, over_x_intervals):
        if pred.size != 0:
            area_pred_over += np.trapz(y=pred, x=ideal) - ideal_area(ideal[-1], ideal[0])
    for pred, ideal in zip(under_intervals, under_x_intervals):
        if pred.size != 0:
            area_pred_under += ideal_area(ideal[-1], ideal[0]) - np.trapz(y=pred, x=ideal)
    area_pred_over /= (100**2)/2
    area_pred_under /= (100**2)/2
    area_pred_over_score = area_pred_over
    area_pred_under_score = area_pred_under
    coverage_deviation_score = (area_pred_over_score + area_pred_under_score)/2

    if get_percentiles:
        return coverage_deviation_score, area_pred_over_score, area_pred_under_score, percentiles, pred_percentiles
    else:
        return coverage_deviation_score, area_pred_over_score, area_pred_under_score



def coverage(y_pred_samples, y_true, perc=95.0, interval_type='ETI', rel_log_errors=False):
    """Whether each sample was covered by predictive interval"""
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()
    within_cred_interval = np.zeros(len(y_true))
    if interval_type == 'ETI':
        '''Equal tailed interval'''
        # lower perc
        q0 = (100.0 - perc) / 2.0
        # upper perc
        q1 = 100.0 - q0
        # print((y_pred_samples[:, 0]).shape)
        for i in range(len(y_true)):
            p0 = np.percentile(y_pred_samples[:, i], q0)
            p1 = np.percentile(y_pred_samples[:, i], q1)
            # if rel_log_errors:
            #     p0, p1 = np.log10(np.exp(1)) *p0/np.mean(y_pred_samples[:, i]), np.log10(np.exp(1)) *p1/np.mean(y_pred_samples[:, i])
            if p0 <= y_true[i] < p1:
                within_cred_interval[i] = 1

            # print("eti", [p0, p1])
    elif interval_type == 'HDI':
        '''Highest density interval'''
        for j in range(len(y_true)):
            print("Interval %s, Datapoints covered: %s percent" % (perc, str(round(j/len(y_true)*100, 2))))
            hdi, x, y, modes = hdi_grid(y_pred_samples[:, j], alpha=1 - perc/100, roundto=10)
            for h in hdi:
                if h[0] <= y_true[j] < h[1]:
                    within_cred_interval[j] = 1
    return np.mean(within_cred_interval)*100

def eti(y_pred_samples, perc=95):
    q0 = (100.0 - perc) / 2.0
    q1 = 100.0 - q0
    etis = np.zeros((2, y_pred_samples.shape[1]))
    for i in range(y_pred_samples.shape[1]):
        p0 = np.percentile(y_pred_samples[:, i], q0)
        p1 = np.percentile(y_pred_samples[:, i], q1)
        etis[0][i] = p0
        etis[1][i] = p1
    # print("ETIS", etis)
    # print("\n")
    return etis


def hdi_grid(sample, alpha=0.05, roundto=2):
    """
    Written by Osvaldo, Martin: https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hdi.py
    Calculate highest posterior density (hdi) of array for given alpha.
    The hdi is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hdi: array with the lower and upper bound

    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    lower = np.min(sample)
    upper = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(lower, upper, 2000)
    y = density.evaluate(x)
    # y = density.evaluate(x, lower, upper) waitting for PR to be accepted
    xy_zipped = zip(x, y / np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1 - alpha):
            break
    hdv.sort()
    diff = (upper - lower) / 20  # differences of 5%
    hdi = [round(min(hdv), roundto)]
    for i in range(1, len(hdv)):
        if hdv[i] - hdv[i - 1] >= diff:
            hdi.append(round(hdv[i - 1], roundto))
            hdi.append(round(hdv[i], roundto))
    hdi.append(round(max(hdv), roundto))
    ite = iter(hdi)
    hdi = list(zip(ite, ite))
    modes = []
    for value in hdi:
        x_hdi = x[(x > value[0]) & (x < value[1])]
        y_hdi = y[(x > value[0]) & (x < value[1])]
        try:
            modes.append(round(x_hdi[np.argmax(y_hdi)], roundto))
        except ValueError:
            # print("WARNING: Did not find mode. None returned for 'modes'. Can be ignored if only hdi limits are of interest.")
            modes.append(None)
    return hdi, x, y, modes
