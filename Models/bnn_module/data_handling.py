import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


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
        data = nan_check(data, printit=False, remove=True)

    data = data.sort_values(by=targets[0])
    data = data.reset_index(drop=True)
    slha = data['file']
    data = data.drop(['file'], axis=1)
    if cut:
        print("\nTargets: %s\nLength of data before preprocess: %s" % (targets, old_data.shape[0]))
        print("Length of data after preprocess: %s\n" % data.shape[0])

    return data, features, targets, slha


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

def data_split(data, target_names, mix_data=None, test_fraction=0.2, seed=42):
    if not isinstance(test_fraction, list):
        fracs = [test_fraction]
    else:
        fracs = test_fraction
    X_ls = []
    y_ls = []
    mix_ls = []
    i = 0
    y = data[target_names]
    X = data.drop(columns=target_names)
    frac_len = len(fracs)
    while i <= frac_len-1:
        frac = fracs[0]
        if frac > 1:
            frac = 1
        X_i = X.sample(frac=frac, random_state=seed, replace=False)
        X_ls.append(X_i)
        y_ls.append(y.loc[X_i.index])
        X = X.drop(X_i.index)
        # fracs = (fracs / (1-fracs[i]))
        if mix_data is not None:
            mix_ls.append(mix_data.loc[X_i.index])
        fracs = fracs[1:]/np.sum(fracs[1:])
        i += 1
    return X_ls, y_ls, mix_ls

def logy_data(df, col_start=0, col_stop=1, inv=False):
    if isinstance(df, pd.DataFrame):
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
        elif inv:
            for i in range(col_start, col_stop):
                column = df.columns[i]
                df[column] = df[column].apply(lambda x: np.exp(x))
            return df
    elif isinstance(df, (np.ndarray, np.generic)):
        if not inv:
            return np.log(df)
        else:
            # for j in df:
            #     if isinstance(j, np.ndarray):
            #         for i in j:
            #             if i > 500:
            #                 print(i)
            return np.exp(df)


def scaling(train_features, test_features, train_target, test_target, scaling='StandardScaler', target_scaling=True):
    columns_features = test_features.columns
    columns_target = test_target.columns
    indicies_train = list(train_features.index.values.tolist())
    indicies_test = list(test_features.index.values.tolist())

    output_scaler = None
    if scaling == 'StandardScaler':
        input_scaler = StandardScaler()
        if target_scaling:
            output_scaler = StandardScaler()
    elif scaling == 'MinMaxScaler':
        input_scaler = MinMaxScaler()
        if target_scaling:
            output_scaler = MinMaxScaler()
    elif scaling == 'RobustScaler':
        input_scaler = RobustScaler()
        if target_scaling:
            output_scaler = RobustScaler()
    else:
        input_scaler = None
        output_scaler = None

    if input_scaler is not None:
        input_scaler.fit(train_features)
        scaled_train_features = input_scaler.transform(train_features)
        scaled_test_features = input_scaler.transform(test_features)

        X_train = pd.DataFrame(scaled_train_features, index=indicies_train, columns=columns_features)
        X_test = pd.DataFrame(scaled_test_features, index=indicies_test, columns=columns_features)
    else:
        X_train = train_features
        X_test = test_features

    if output_scaler is not None:
        output_scaler.fit(train_target)

        scaled_train_target = output_scaler.transform(train_target)
        scaled_test_target = output_scaler.transform(test_target)
        y_train = pd.DataFrame(scaled_train_target, index=indicies_train, columns=columns_target)
        y_test = pd.DataFrame(scaled_test_target, index=indicies_test, columns=columns_target)

    else:
        y_train = train_target
        y_test = test_target

    return X_train, X_test, y_train, y_test, output_scaler, input_scaler


def scaling_specific(X, inverse=False, scaler='MinMaxScaler', scaler_obj=None):
    if not inverse:
        if scaler == 'MinMaxScaler':
            if scaler_obj is None:
                scaler_obj = MinMaxScaler()
                scaler_obj.fit(X)
                X = scaler_obj.transform(X).ravel()
                return X, scaler_obj
            else:
                scaler_obj.fit(X)
                X = scaler_obj.transform(X).ravel()
                return X
        else:
            print("No scaler")
    elif inverse and scaler_obj is not None:
        if isinstance(X, pd.DataFrame):
            X[X.columns] = scaler_obj.inverse_transform(X[X.columns])
        else:
            X = scaler_obj.inverse_transform(X.reshape(-1, 1)).reshape(X.shape)
        return X


def inverse_affine_scaling(data, scaler_obj=None, is_input=True, is_variance=False):
    if scaler_obj is None:
        print("Set the scaler objects used on the data")
        return
    else:
        scaler_name = str(scaler_obj).split('(')[0]

    if is_input:
        # If input
        return inverse_scaling_list(data, scaler_obj)
    else:
        # If output
        if not is_variance:
            return inverse_scaling_list(data, scaler_obj)
        elif is_variance:
            if scaler_name == 'StandardScaler':
                scaling_factor = scaler_obj.var_[0]
            elif scaler_name == 'MinMaxScaler':
                scaling_factor = (1 / scaler_obj.scale_[0]) ** 2

            y_vars = data
            y_vars = np.squeeze(y_vars)
            if isinstance(y_vars[0], (list, pd.Series, np.ndarray)):
                for i in range(len(y_vars)):
                    y_vars[i] *= scaling_factor
            else:
                y_vars *= scaling_factor
            return y_vars
        else:
            print("is_variance must have boolean value")

def uncertainties(model, y_pred_samples, y_pred_var_samples, inv=True, std_multiplier=1):
    # inv=True transforms back the log transformed variances to geometric standard deviations of the log_normal distribution.
    # y_sigma_aleo is the geometric aleoteric uncertainty. It is the result of transforming the mean of sample variances of the predictive distribution, and taking the root.
    # y_sigma_epis is the geometric epistemic uncertainty. It is the result of transforming the standard deviations of the sample predictions of the pred. dist, and taking the root.
    # All of the above is for each data point respectively. y_sigma_mix is the root of the squared sum of uncertainties: sqrt(aleo² + epis²), it is equivalent to the standard deviation of
    # a gaussian log-mixture distribution.
    y_sigma_aleo, y_sigma_epis, y_sigma_mix = model.predict_sigmas(y_pred_samples, y_pred_var_samples,
                                                                   inv=inv,
                                                                   n=std_multiplier)
    return y_sigma_aleo, y_sigma_epis, y_sigma_mix

def inverse_log_scaling(y_ls):
    # Data only. For predictions it must be used after producing the log-normal uncertainties
    return inverse_scaling_list(y_ls, invert_log=True)

def inverse_scaling_list(quantities, scaler=None, invert_log=False):
    # Loops through list of arrays or dataframes and inverts.
    if not invert_log:
        if not isinstance(quantities, list):
            quantities = [quantities]
        new_quantities = []
        for q in quantities:
            new_quantities.append(np.reshape(scaling_specific(q, inverse=True, scaler_obj=scaler), np.shape(q)))
    elif invert_log:
        if not isinstance(quantities, list):
            quantities = [quantities]
        new_quantities = []
        for q in quantities:
            new_quantities.append(logy_data(q, inv=True))
    return new_quantities

def insertion_sort(ls, sort_by='batch_size'):
    if sort_by == 'layers':
        for i in range(len(ls)):
            cursor = ls[i]
            pos = i

            while pos > 0 and sum(ls[pos - 1][sort_by]) > sum(cursor[sort_by]):
                ls[pos] = ls[pos - 1]
                pos = pos - 1
            ls[pos] = cursor
    else:
        for i in range(len(ls)):
            cursor = ls[i]
            pos = i

            while pos > 0 and ls[pos-1][sort_by] > cursor[sort_by]:
                ls[pos] = ls[pos - 1]
                pos = pos - 1
            ls[pos] = cursor
    return ls
