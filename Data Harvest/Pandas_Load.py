import numpy as np
import pandas as pd
import os
import os.path
from os import path

"""
using load

First title, e.g. EwOnly
Second, P: parameters, M: masses, O: other relevant masses,  X: Mixture, 
Third, CS: Cross section data

Underscore needed after title.

Example: load(EwOnly_PMOX_CS)
or 
Example: load(EwOnly_PMOXCS)
"""

def write_slha(filename, directory, include):
    if include is not None:
        print('Running command:')
        print("python harvest_slha_ew.py %s %s ''" % (filename, directory))
        os.putenv("include", include)
        os.system("python NN_EW_modules/harvest_slha_ew.py %s %s ''" % (filename, directory))
    elif include is None:
        print("python harvest_slha_ew.py %s %s '' ''" % (filename, directory))
        os.system("python NN_EW_modules/harvest_slha_ew.py %s %s ''" % (filename, directory))


def load(init, include=None, directory='/test_data/EWonly/', rw=False, title_append='',
         abs_parameters=False, abs_masses=True, return_df=True):
    first_title = directory.split('/')[-2]
    filename = first_title + '_' + init + '_' + title_append

    if abs_parameters: os.putenv("abs_p", "True")
    if abs_masses: os.putenv("abs_m", "True")

    if path.exists(filename) and include is None and rw is False:
        print('Using existing file \n')

    elif path.exists(filename) and include is None and rw is True:
        print('Replacing existing file.\n')
        write_slha(filename, directory, include)

    elif path.exists(filename) and include is not None and rw is True:
        print('Replacing existing file.')
        print('Where specified, including only given elements. Where not specified, including all elements.')
        print('Using inclusion string "%s"\n' % (include))
        write_slha(filename, directory, include)

    elif not path.exists(filename):
        print('Creating new file %s' % filename)
        print('Where specified, including only given elements. Where not specified, including all elements.')
        print('Using inclusion string %s\n' % (include))
        write_slha(filename, directory, include)

    if return_df:
        df = pd.read_csv(filename, sep=" ", skipinitialspace=True)
        df.head()
        df = df.iloc[:, :-1]
        return df, filename
    else:
        return filename


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
                df[column] = np.log10(df[column])
            except (ValueError, AttributeError):
                pass
            return df
    if inv:
        for i in range(col_start, col_stop):
            column = df.columns[i]
            try:
                df[column] = df[column].pow(10)
            except (ValueError, AttributeError):
                pass
            return df


