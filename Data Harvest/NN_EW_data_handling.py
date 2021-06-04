import os
import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from NN_EW_modules import Pandas_Load as pl
from NN_EW_modules import NN_EW_helper_functions as hf

def get_mixing_df(data):
    mix_df = pd.DataFrame({'bino': np.square(data['nmix11']),
                           'wino': np.square(data['nmix12']),
                           'higgsino': np.square(data['nmix13']) + np.square(data['nmix14'])},
                          columns=['bino', 'wino', 'higgsino'])

    mix_df['mixing max'] = np.argmax(mix_df.values, axis=-1)
    mix_df['mixing max'] = mix_df['mixing max'].astype(str)
    mix_df['mixing max'].replace(['0'], 'bino', inplace=True)
    mix_df['mixing max'].replace(['1'], 'wino', inplace=True)
    mix_df['mixing max'].replace(['2'], 'higgsino', inplace=True)

    return mix_df


def names_creation(targets, load_stored=False):
    # Creating a filename, using given targets.
    save_string = 'Final_model'
    file_append_string = ''
    for t in targets:
        t = t.replace("0000", 'ø')
        t = t.split('_')
        t1 = t[0]
        t2 = t[1]
        save_string += '_' + t1 + '-' + t2
        file_append_string += t1 + '-' + t2
    # Checking if there exists a stored model for the given filename.
    if load_stored:
        if not (os.path.exists(save_string)):
            print('No stored model')
            load_stored = False

    # Inferring the mass names from the names of the given cross section targets.
    # Masses corresponding to the cross section names must be included
    masses = []
    for t in targets:
        mass1 = t.split('_')[0]
        mass1 = 'm' + mass1

        mass2 = t.split('_')[1]
        mass2 = 'm' + mass2

        masses.append(mass1)
        masses.append(mass2)
    masses = list(dict.fromkeys(masses))

    return masses, save_string, load_stored

def get_data(targets_configs, directory, rewrite=False, return_df=True, get_params='PMCSX'):
    mass_string = 'M: '
    cs_string = 'CS: '
    masses_all = []
    targets_all = []
    for targets in targets_configs:
        masses, _, _ = names_creation(targets, load_stored=False)
        masses_all += masses
        targets_all += targets
    masses_all = list(dict.fromkeys(masses_all))
    targets_all = list(dict.fromkeys(targets_all))

    for target in targets_all:
        cs_string += 'm' + target + ' '
    for mass in masses_all:
        mass_string += mass + ' '
    cs_string = cs_string.replace("_13000_NLO_1", '')
    cs_string = cs_string.replace("_", '_m')

    string_title_append = cs_string.replace("10000", '')
    string_title_append = string_title_append.replace("m", '')
    string_title_append = string_title_append.replace("_13000_NLO_1", '')
    string_title_append = string_title_append.replace("CS: ", '')
    string_title_append = string_title_append.replace("_", '-')
    string_title_append = string_title_append.replace(" ", '_')
    string_title_append = string_title_append[:-1]

    filename, data = pl.load(get_params, directory=directory, title_append=string_title_append, rw=rewrite,
                   include=mass_string + cs_string, abs_masses=False,
                   abs_parameters=False, return_df=return_df)
    if return_df:
        return data, filename
    else:
        return filename

def preprocess_remover(data, targets):
    for i in range(len(targets)):
        data = data.drop(data[(data[targets[i]] == -1)].index)
        data = hf.nan_check(data, printit=False, remove=True)
    return data


def preprocess(data, targets, masses, mixing=True, cut=True):
    old_data = data
    if mixing:
        features = list(data.columns)
        features = [x for x in features if "NLO" not in x]
        features = [x for x in features if "0000" not in x]
        for m in masses:
            features.append(m)
        keep = features + targets
        keep_fin = []
        for k in keep:
            if k in data.columns:
                keep_fin.append(k)
        data = data[keep_fin]
    else:
        features = ["tanb", "M_1(MX)", "M_2(MX)", "mu(MX)", "m1000022"]
        targets = ["1000022_1000022_13000_NLO_1"]
        print("\nOnly m1ø22 available, so no mixing included")

    if cut:
        for i in range(len(targets)):
            data = data.drop(data[(data[targets[i]] == -1)].index)
        data = hf.nan_check(data, printit=False, remove=True)

    data = data.sort_values(by=targets[0])
    data = data.reset_index(drop=True)
    slha = data['file']
    data = data.drop(['file'], axis=1)
    if cut:
        print("\nTargets: %s\nLength of data before preprocess: %s" % (targets, old_data.shape[0]))
        print("Length of data after preprocess: %s\n" % data.shape[0])

    return data, features, targets, slha


def data_split(data, target_names, mix_data=None, fraction=0.8, seed=42):

    train_data = data.sample(frac=fraction, random_state=seed)
    test_data = data.drop(train_data.index)
    if mix_data is not None:
        mix_data_test = mix_data.drop(train_data.index)

    train_target = train_data[target_names]
    train_data.drop(columns=target_names, inplace=True)

    test_target = test_data[target_names]
    test_data.drop(columns=target_names, inplace=True)

    if mix_data is None:
        return train_data, test_data, train_target, test_target
    else:
        return train_data, test_data, train_target, test_target, mix_data_test

def scaling(train_data, test_data, train_target, test_target, scaling='StandardScaler'):
    columns_features = test_data.columns
    columns_target = test_target.columns
    indicies_train = list(train_data.index.values.tolist())
    indicies_test = list(test_data.index.values.tolist())

    if scaling == 'StandardScaler':
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()
    elif scaling == 'MinMaxScaler':
        input_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()
    elif scaling == 'RobustScaler':
        input_scaler = RobustScaler()
        output_scaler = RobustScaler()
    else:
        input_scaler = None
        output_scaler = None

    if input_scaler is not None:
        input_scaler.fit(train_data)
        train_t_inp = input_scaler.transform(train_data)
        test_t_inp = input_scaler.transform(test_data)

        scaled_train_data = pd.DataFrame(train_t_inp, index=indicies_train,  columns=columns_features)
        scaled_test_data = pd.DataFrame(test_t_inp, index=indicies_test, columns=columns_features)
    else:
        scaled_train_data = train_data
        scaled_test_data = test_data

    if output_scaler is not None:
        output_scaler.fit(train_target.values.reshape(-1, 1))

        train_t_outp = output_scaler.transform(train_target)
        test_t_outp = output_scaler.transform(test_target)
        scaled_train_target = pd.DataFrame(train_t_outp, index=indicies_train, columns=columns_target)
        scaled_test_target = pd.DataFrame(test_t_outp, index=indicies_test, columns=columns_target)

    else:
        scaled_train_target = train_target
        scaled_test_target = test_target

    return scaled_train_data, scaled_test_data, scaled_train_target, scaled_test_target, output_scaler, input_scaler

def create_df_pred_true(y_pred, y_true, mix_df_test, slha=None):
    if slha is None:
        slha = y_true.iloc[:, 0]
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.drop(['file'], axis=1).values.ravel()
        y_true = y_true.drop(['file'], axis=1).values.ravel()
    df_pt = pd.DataFrame({'file': slha,
                          'test_target': y_true,
                          'test_predictions': y_pred,
                          'bino': mix_df_test['bino'],
                          'wino': mix_df_test['wino'],
                          'higgsino': mix_df_test['higgsino'],
                          'mixing max': mix_df_test['mixing max']},
                         columns=['file', 'test_target', 'test_predictions', 'bino', 'wino', 'higgsino', 'mixing max'])
    return df_pt
