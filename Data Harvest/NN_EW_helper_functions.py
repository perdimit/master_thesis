import os
import os.path
import shutil
import sys
import subprocess
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

"""log of data"""
def logy_data(df, col_start=0, col_stop=1, inv=False):
    if not 0 <= col_start < len(df.columns) and col_start < col_stop <= len(df.columns):
        print('range error. Log only first column')
        col_start = 0
        col_stop = 1
    if not inv:
        for i in range(col_start, col_stop):
            column = df.columns[i]
            df = df.drop(df[(df[column] <= 0)].index)
            try:
                df[column] = np.log(df[column])
            except (ValueError, AttributeError):
                pass
        return df
    if inv:
        for i in range(col_start, col_stop):
            column = df.columns[i]
            df[column] = df[column].apply(lambda x: np.exp(x))
            # try:
            #     print("here")
            #     print(df[column])
            # except (ValueError, AttributeError):
            #     pass
        return df


def adj_r2(y_true, y_pred, data):
    r2 = r2_score(y_true, y_pred)
    adjr2 = (1 - (1 - r2) * ((data.shape[0] - 1) / (data.shape[0] - data.shape[1] - 1)))
    return adjr2


def nan_check(df_check, printit=False, remove=True, col_start=1):
    for c in range(col_start, len(df_check.columns)):
        index_nan = list(df_check[df_check.columns[c]].index[df_check[df_check.columns[c]].apply(np.isnan)].copy())
        index_nan.sort()
        for i in index_nan:
            if printit:
                print(df_check.loc[[i]])
                print("")
            if remove:
                df_check = df_check.drop(index=i)
    return df_check

def metrics_to_txtfile(metrics, metrics_name):
    spaces = ' '*4
    new_columns = spaces.join(metrics.columns)
    m = metrics.to_csv(sep='%', header=None, encoding='utf-8', mode='a', quotechar='\a')

    m_new = ''
    check = 0
    count = 0
    count2 = 0
    for s in m:
        if s == '\n':
            check = 0
            count = 0
            count2 = 0
        # first space(after MSE etc)
        if s == '%' and check == 0:
            room = 8 - count
            m_new += ' '*room
            check = 1

        # second space(after first value)
        elif s == '%' and check == 1:
            room2 = 7 - count2
            m_new += ' '*(10+room2)
            if s in {'%', ' '}:
                count2 = 0
            check = 2

        # all consecutive spaces
        elif s == '%' and check in {2, 3}:
            room2 = 7 - count2
            # room2 = 0
            m_new += ' ' * (24 + room2)
            if s in {'%', ' '}:
                count2 = 0
            check = 3
        # All characters
        elif s != '%':
            count += 1
            if check in {1, 2, 3} and s != ' ':
                count2 += 1
            if s == '\n':
                count -= 1
            m_new += s

    with open(metrics_name, "a") as file:
        file.write('\t')
        file.write(new_columns)
        file.write('\n')
        file.write(m_new)
        file.write("\n")

def rel_error(true, pred, sd=None):
    return (true - pred) / true


def error_sd(true, pred, sd=None):
    if sd is None:
        return (true - pred) / np.std(pred)
    if sd is not None:
        return (true - pred) / sd

def stack_outliers(x, sd_prod=2):
    mean = np.mean(x)
    sd = np.std(x)

    lim_neg = (-sd_prod*sd).values[0]
    lim_pos = (+sd_prod*sd).values[0]

    ind_neg = x.index[x['error'] <= lim_neg].tolist()
    ind_pos = x.index[x['error'] >= lim_pos].tolist()

    x.loc[x['error'] <= lim_neg, 'error'] = lim_neg
    x.loc[x['error'] >= lim_pos, 'error'] = lim_pos

    return x, ind_neg, ind_pos, mean.values[0], sd.values[0]

