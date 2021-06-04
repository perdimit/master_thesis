import os
import os.path
import shutil
import sys
import subprocess
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# must be installed: pip install git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px
import plotly.figure_factory as ff

from NN_EW_modules import NN_EW_helper_functions as hf
from NN_EW_modules import NN_EW_data_handling as dh

#################
"""Build model"""


#################


def build_model(act_func, data, target, hidden_layers, nodes):
    if act_func not in ['sigmoid', 'relu', 'elu', 'selu']:
        return None
    model = keras.Sequential([
        layers.Dense(nodes, activation=act_func, input_shape=[len(data.keys())])
    ])

    for i in range(hidden_layers):
        model.add(layers.Dense(nodes, activation=act_func))
    model.add(layers.Dense(target.shape[1]))

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model


"""Train model"""


def train_model(model, scaled_train_data, scaled_train_target, nn_it, save_filename, save_model=True, load_stored=False,
                patience=100, loss_plot=None):
    EPOCHS = 10000
    b_size = 32
    # creates checkpoint if the session is terminated for some reason
    checkpoint_dir0 = "training_"
    checkpoint_dir = checkpoint_dir0 + str(nn_it)

    # checks if there's an interrupted session, and if so loads the earlier iterations and starts from the right iteration.
    for x in os.listdir('.'):
        if x.startswith(checkpoint_dir0):
            interrupt_it = x.split('_')[1]
            if nn_it < int(interrupt_it):
                load_stored = True

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    NaN_callback = keras.callbacks.TerminateOnNaN()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                     monitor='val_loss',
                                                     save_best_only=False,
                                                     save_weights_only=False,
                                                     mode='min',
                                                     verbose=0,
                                                     period=100)

    # Remove this one when epochs are stored by json
    used_checkpoint = False
    if (os.path.exists(checkpoint_dir + '/saved_model.pb')) and not load_stored:
        print("Last training session was interrupted. Continuing from last checkpoint")
        used_checkpoint = True
        model = tf.keras.models.load_model(checkpoint_dir)

        history = model.fit(
            scaled_train_data, scaled_train_target, batch_size=b_size,
            epochs=EPOCHS, validation_split=0.2, verbose=0,
            callbacks=[early_stop, tfdocs.modeling.EpochDots(), cp_callback, NaN_callback])
    elif (os.path.exists(save_filename)) and load_stored:
        print("Loading stored model at %s" % save_filename)
        model = tf.keras.models.load_model(save_filename)
    else:
        print("Starting training session of neural network %s" % nn_it)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        history = model.fit(
            scaled_train_data, scaled_train_target, batch_size=b_size,
            epochs=EPOCHS, validation_split=0.2, verbose=0,
            callbacks=[early_stop, tfdocs.modeling.EpochDots(), cp_callback, NaN_callback])

    # Saving final model
    if save_model and not load_stored:
        model.save(save_filename)
        print("\nModel saved at %s" % save_filename)
    # deleting checkpoint files
    if not load_stored:
        open(checkpoint_dir + '/saved_model.pb', 'w').close()
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    if loss_plot is not None:
        if not used_checkpoint and not load_stored:
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            hist.to_csv(save_filename + '/loss_history.csv', index=False)
            loss_plot.plot(hist['epoch'], hist['mse'], color='#1f77b4', label='Train MSE')
            loss_plot.plot(hist['epoch'], hist['val_mse'], linestyle='dashed', color='#1f77b4', label='Validation MSE')
            loss_plot.grid()
            loss_plot.legend(loc="upper right")
            # plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
            # plotter.plot({'Basic': history}, metric="mse")
            loss_plot.set_ylabel('MSE')
            loss_plot.set_xlabel('Epoch')
        elif not used_checkpoint and load_stored and os.path.exists(save_filename + '/loss_history.csv'):
            hist = pd.read_csv(save_filename + '/loss_history.csv')
            # currently not smoothed
            loss_plot.plot(hist['epoch'], hist['mse'], color='#1f77b4', label='Train MSE')
            loss_plot.plot(hist['epoch'], hist['val_mse'], linestyle='dashed', color='#1f77b4', label='Validation MSE')
            loss_plot.legend(loc="upper right")
            loss_plot.grid()
            loss_plot.set_ylabel('MSE')
            loss_plot.set_xlabel('Epoch')

    return model, loss_plot


# Return prediction and true values inversed scaled.
def pred_and_true(model, scaled_test_data, scaled_test_target, output_scaler, slha):
    test_predictions = model.predict(scaled_test_data)
    test_predictions = pd.DataFrame(test_predictions, columns=scaled_test_target.columns)
    y_pred = output_scaler.inverse_transform(test_predictions)
    y_true = output_scaler.inverse_transform(scaled_test_target)
    indicies = list(scaled_test_target.index.values.tolist())
    y_pred = pd.DataFrame(y_pred, index=indicies, columns=scaled_test_target.columns)
    y_true = pd.DataFrame(y_true, index=indicies, columns=scaled_test_target.columns)
    slha_y = slha.loc[indicies]
    y_pred.insert(0, 'file', slha_y)
    y_true.insert(0, 'file', slha_y)
    return y_pred, y_true


def pred_and_true_mix(mix_df_test, y_pred, y_true):
    mix_fetchindx = mix_df_test.reset_index(drop=True)
    bino_i = mix_fetchindx.index[mix_fetchindx['mixing max'] == 'bino'].tolist()
    wino_i = mix_fetchindx.index[mix_fetchindx['mixing max'] == 'wino'].tolist()
    higgsino_i = mix_fetchindx.index[mix_fetchindx['mixing max'] == 'higgsino'].tolist()
    list_i = [bino_i, wino_i, higgsino_i]

    y_pred_b = np.zeros((len(bino_i)))
    y_pred_w = np.zeros((len(wino_i)))
    y_pred_h = np.zeros((len(higgsino_i)))
    y_pred_ls = np.array([y_pred_b, y_pred_w, y_pred_h])

    y_true_b = np.zeros((len(bino_i)))
    y_true_w = np.zeros((len(wino_i)))
    y_true_h = np.zeros((len(higgsino_i)))
    y_true_ls = np.array([y_true_b, y_true_w, y_true_h])

    for m in range(len(list_i)):
        for i in range(len(list_i[m])):
            y_pred_ls[m][i] = y_pred[list_i[m][i]]
            y_true_ls[m][i] = y_true[list_i[m][i]]
    return y_pred_ls, y_true_ls


"""Plots"""


def add_plot(fig, y_true, y_pred, mix_df_test, slha, r, c, i, target_name, mixing_info=False):
    if r == 0 and c == 0:
        legend_bool = True
    else:
        legend_bool = False
    min_lim = np.min([np.min(y_true), np.min(y_pred)])
    max_lim = np.max([np.max(y_true), np.max(y_pred)])
    lims = [min_lim, max_lim]

    lin_reg = LinearRegression()
    lin_reg_bino = LinearRegression()
    lin_reg_wino = LinearRegression()
    lin_reg_higgsino = LinearRegression()

    y_pred_ls, y_true_ls = pred_and_true_mix(mix_df_test, y_pred, y_true)

    slope = lin_reg.fit(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
    slope_bino = lin_reg_bino.fit(y_true_ls[0].reshape(-1, 1), y_pred_ls[0].reshape(-1, 1))
    slope_wino = lin_reg_wino.fit(y_true_ls[1].reshape(-1, 1), y_pred_ls[1].reshape(-1, 1))
    slope_higgsino = lin_reg_higgsino.fit(y_true_ls[2].reshape(-1, 1), y_pred_ls[2].reshape(-1, 1))

    x_reg = np.linspace(lims[0], lims[1], len(y_true)).ravel()
    x_reg_bino = np.linspace(lims[0], lims[1], len(y_true_ls[0])).ravel()
    x_reg_wino = np.linspace(lims[0], lims[1], len(y_true_ls[1])).ravel()
    x_reg_higgsino = np.linspace(lims[0], lims[1], len(y_true_ls[2])).ravel()

    y_reg = slope.predict(x_reg.reshape(-1, 1)).ravel()
    y_reg_bino = slope_bino.predict(x_reg_bino.reshape(-1, 1)).ravel()
    y_reg_wino = slope_wino.predict(x_reg_wino.reshape(-1, 1)).ravel()
    y_reg_higgsino = slope_higgsino.predict(x_reg_higgsino.reshape(-1, 1)).ravel()
    df_pt = dh.create_df_pred_true(y_pred, y_true, mix_df_test, slha)
    df_pt['x_regression'] = x_reg
    df_pt['y_regression'] = y_reg

    color = [
        0 if m == 'bino' else 1 if m == 'wino' else 2
        for m in df_pt['mixing max']
    ]
    colorscale = [[0, 'rgb(214, 39, 40)'], [0.5, 'mediumaquamarine'], [1, 'mediumslateblue']]

    if not mixing_info:
        fig.add_trace(
            go.Scattergl(x=df_pt['x_regression'], y=df_pt['y_regression'], line=dict(color='black', width=1),
                         name='Regression', hoverinfo="none",
                         showlegend=legend_bool),
            row=r + 1, col=c + 1)

    fig.add_trace(go.Scattergl(x=df_pt['test_target'], y=df_pt['test_predictions'], mode='markers', name='True/pred',
                               showlegend=False,
                               marker={'color': color,
                                       'colorscale': colorscale,
                                       'size': 3.5,
                                       'opacity': 0.8
                                       },
                               text=['file: %s<br>bino: %s<br>wino: %s<br>higgsino: %s' % (s, b, w, h) for s, b, w, h in
                                     df_pt.loc[:, ['file', 'bino', 'wino', 'higgsino']].values], hoverinfo='text'),
                  row=r + 1, col=c + 1)

    if mixing_info:
        group_var = str(i)
        fig.add_trace(go.Scattergl(x=[None], y=[None], mode='markers',
                                   marker=dict(size=0.1, color='white'), legendgroup=group_var,
                                   showlegend=True, name=target_name + ':'))
        R2_reg = round(r2_score(y_pred, y_true), 4)
        fig.add_trace(
            go.Scattergl(x=df_pt['x_regression'], y=df_pt['y_regression'], line=dict(color='black', width=2),
                         name='Regression, R2: %s' % R2_reg, hoverinfo="none",
                         showlegend=True, legendgroup=group_var),
            row=r + 1, col=c + 1)

        R2_reg_bino = round(r2_score(y_pred_ls[0], y_true_ls[0]), 4)
        fig.add_trace(
            go.Scattergl(x=x_reg_bino, y=y_reg_bino,
                         line=dict(color=colorscale[0][1], width=3, dash='dash'),
                         name='Regression Bino R2: %s' % R2_reg_bino, hoverinfo="none",
                         showlegend=True, legendgroup=group_var),
            row=r + 1, col=c + 1)

        R2_reg_wino = round(r2_score(y_pred_ls[1], y_true_ls[1]), 4)
        fig.add_trace(
            go.Scattergl(x=x_reg_wino, y=y_reg_wino,
                         line=dict(color=colorscale[1][1], width=3, dash='dash'),
                         name='Regression Wino R2: %s' % R2_reg_wino, hoverinfo="none",
                         showlegend=True, legendgroup=group_var),
            row=r + 1, col=c + 1)

        R2_reg_higgsino = round(r2_score(y_pred_ls[2], y_true_ls[2]), 4)
        fig.add_trace(
            go.Scattergl(x=x_reg_higgsino, y=y_reg_higgsino,
                         line=dict(color=colorscale[2][1], width=3, dash='dash'),
                         name='Regression Higgsino R2: %s' % R2_reg_higgsino, hoverinfo="none",
                         showlegend=True, legendgroup=group_var),
            row=r + 1, col=c + 1)


def pred_and_true_plot(y_pred, y_true, mix_df_test, target_names, mixing_info=False, error_plot=True, denom_sd=True,
                       type_plot='probability', sd_product=0.1, bin_size=0.001):
    slha = y_true.iloc[:, 0]
    y_pred = y_pred.drop(['file'], axis=1).values
    y_true = y_true.drop(['file'], axis=1).values

    target_len = y_true.shape[1]
    if error_plot:
        rows_num = target_len
        cols_num = 2
    else:
        cols = 2
        rows_num = math.ceil(target_len / cols)
        if target_len >= cols:
            cols_num = cols
        else:
            cols_num = target_len

    # Line from corner to corner instead of shared_xaxes etc
    ideal_min = np.min([np.min(y_true), np.min(y_pred)])
    ideal_max = np.max([np.max(y_true), np.max(y_pred)])
    ideal_lims = [ideal_min - 0.5, ideal_max + 0.5]
    ideal_line = np.linspace(ideal_lims[0], ideal_lims[1], 100).ravel()

    if error_plot:
        err_specs = []
        for i in range(rows_num):
            err_specs.append([{"rowspan": 2}, {}])
            err_specs.append([None, {}])

        tn_extend = []
        for i in range(len(target_names)):
            tn_extend.append(target_names[i])
            tn_extend.append('')
            tn_extend.append('')
        fig = make_subplots(
            rows=rows_num * 2, cols=cols_num, specs=err_specs,
            print_grid=False, subplot_titles=tn_extend, vertical_spacing=0.04, horizontal_spacing=0.065)
    else:
        fig = make_subplots(rows=rows_num, cols=cols_num, subplot_titles=target_names, vertical_spacing=0.05,
                            shared_xaxes=True, shared_yaxes=True)

    leg_bool = True
    i = 0

    for r in range(rows_num):
        if error_plot:

            leg_bool = False

            if type_plot == 'probability':
                type_plot_name = 'Probability'
                curve_bool = True
                ym = 0.01

            else:
                type_plot = ''
                type_plot_name = 'Frequency'
                curve_bool = False
                ym = 30
            fig.add_trace(go.Scatter(x=[0, 0],
                                     y=[0, ym],
                                     mode="lines",
                                     legendgroup="a",
                                     showlegend=False, opacity=0.5,
                                     marker=dict(size=12,
                                                 line=dict(width=0.01),
                                                 color="black"
                                                 )), row=2 * r + 1, col=2
                          )

            yp_10 = np.power(y_pred[:, i], 10)
            yt_10 = np.power(y_true[:, i], 10)
            fig_dist = error_plots(yp_10, yt_10, mix_df_test, target_names[r], slha,
                                   denom_sd=denom_sd, bin_size=bin_size, type_plot=type_plot,
                                   curve_bool=curve_bool, sd_product=sd_product)

            distplot = fig_dist['data']

            for dp in range(12):

                if type_plot == 'probability':
                    if i > 0:
                        distplot[dp]['showlegend'] = False
                    if dp < 8:
                        fig.add_trace(distplot[dp], 2 * r + 1, 2)
                    elif dp >= 8:
                        fig.add_trace(distplot[dp], 2 * r + 2, 2)
                else:
                    if i > 0 and dp < 8:
                        distplot[dp]['showlegend'] = False
                    if dp < 4:
                        fig.add_trace(distplot[dp], 2 * r + 1, 2)
                    elif dp < 8:
                        fig.add_trace(distplot[dp], 2 * r + 2, 2)

            fig.add_trace(
                go.Scattergl(x=ideal_line, y=ideal_line, line=dict(color='grey', width=1, dash='longdash'),
                             name='Ideal line', hoverinfo="none", legendgroup='mixing',
                             showlegend=leg_bool), row=2 * r + 1, col=1)

            add_plot(fig, y_true[:, i], y_pred[:, i], mix_df_test, slha, 2 * r, 0, i, target_names[i],
                     mixing_info=mixing_info)

            # Update xaxis properties
            axis_txt_size = 12
            if error_plot and denom_sd:
                fig.update_xaxes(title_text=r"${\Large \frac{y_{true}-y_{pred}}{\sigma_{pred}}}$", row=2 * r + 1,
                                 col=2, title_font=dict(size=axis_txt_size))
                fig.update_xaxes(title_text=r"${\Large \frac{y_{true}-y_{pred}}{\sigma_{pred}}}$", row=2 * r + 2,
                                 col=2, title_font=dict(size=axis_txt_size))

            elif error_plot and not denom_sd:
                fig.update_xaxes(title_text=r"${\Large \frac{y_{true} - y_{pred}}{y_{true}}}$", row=2 * r + 1,
                                 col=2, title_font=dict(size=axis_txt_size))
                fig.update_xaxes(title_text=r"${\Large \frac{y_{true} - y_{pred}}{y_{true}}}$", row=2 * r + 2,
                                 col=2, title_font=dict(size=axis_txt_size))

            if error_plot:
                fig.update_yaxes(title_text=type_plot_name, row=2 * r + 1, col=2,
                                 title_font=dict(size=axis_txt_size))
                fig.update_yaxes(title_text="Predicted", row=2 * r + 1, col=1, title_font=dict(size=axis_txt_size))
                fig.update_xaxes(title_text="True", row=2 * r + 1, col=1, title_font=dict(size=axis_txt_size))

            i += 1

        else:
            for c in range(cols_num):
                if i < target_len:
                    if i > 0:
                        leg_bool = False
                    fig.add_trace(
                        go.Scattergl(x=ideal_line, y=ideal_line, line=dict(color='grey', width=1, dash='longdash'),
                                     name='Ideal line', hoverinfo="none", legendgroup='mixing',
                                     showlegend=leg_bool), row=r + 1, col=c + 1)

                    add_plot(fig, y_true[:, i], y_pred[:, i], mix_df_test, slha, r, c, i, target_names[i],
                             mixing_info=mixing_info)
                    i += 1

    fig.update_layout(
        barmode='overlay')
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="#7f7f7f"
        )
    )
    if target_len > 1:
        fig.update_layout(autosize=False,
                          width=1780,
                          height=2750)

        fig.update_layout(
            margin=dict(l=20, r=20, t=55, b=55)
        )

    fig.show(config={"displayModeBar": False, "showTips": False})


def error_plots(y_pred, y_true, mix_df_test, target_name, slha=None, bin_size=0.025, denom_sd=True, type_plot='', curve_bool=True,
                rug_bool=True, sd_product=0.1):
    df_pt = dh.create_df_pred_true(y_pred, y_true, mix_df_test, slha)
    df_bino = df_pt[df_pt['mixing max'] == 'bino']
    df_wino = df_pt[df_pt['mixing max'] == 'wino']
    df_higgsino = df_pt[df_pt['mixing max'] == 'higgsino']

    if denom_sd:
        error = getattr(hf, 'error_sd')
    elif not denom_sd:
        error = getattr(hf, 'rel_error')

    sd = np.std(df_pt['test_predictions'])
    x1 = pd.DataFrame(error(df_pt['test_target'], df_pt['test_predictions']), columns=['error'])
    x1['mixing max'] = df_pt['mixing max']

    x1, ind_neg, ind_pos, err_mean, err_sd = hf.stack_outliers(x1, sd_prod=sd_product)

    x2 = error(df_bino['test_target'], df_bino['test_predictions'], sd)
    x3 = error(df_wino['test_target'], df_wino['test_predictions'], sd)
    x4 = error(df_higgsino['test_target'], df_higgsino['test_predictions'], sd)

    neg_lim = - sd_product * err_sd
    pos_lim = + sd_product * err_sd

    if type_plot == 'probability':
        type_plot_str = 'Probability'
    elif type_plot == '':
        type_plot_str = 'Frequency'

    print("\nHistogram info %s:" % target_name)
    print("Plot type: %s" % type_plot_str)
    print("Limit negative side: ", neg_lim)
    print("Limit positive side: ", pos_lim)
    print("sigma_{pred} = ", np.std(df_pt['test_predictions']))

    for i in ind_neg:
        mixing = x1['mixing max'].loc[i]
        if mixing == 'bino':
            x2.loc[i] = neg_lim
        if mixing == 'wino':
            x3.loc[i] = neg_lim
        if mixing == 'higgsino':
            x4.loc[i] = neg_lim

    for i in ind_pos:
        mixing = x1['mixing max'].loc[i]
        if mixing == 'bino':
            x2.loc[i] = pos_lim
        if mixing == 'wino':
            x3.loc[i] = pos_lim
        if mixing == 'higgsino':
            x4.loc[i] = pos_lim

    hist_data = [x1['error'], x2, x3, x4]

    group_labels = ['Total', 'Bino', 'Wino', 'Higgsino']

    colorscale = ['orange', 'rgb(214, 39, 40)', 'mediumaquamarine', 'mediumslateblue']

    fig = ff.create_distplot(hist_data, group_labels=group_labels, bin_size=bin_size,
                             colors=colorscale, curve_type='kde', histnorm=type_plot, show_curve=curve_bool,
                             show_rug=rug_bool)

    return fig


"""Metrics"""


def metrics(data, target, pred, target_names, mix_df_test=None, mixing_info=False):
    rounding = 5
    target = target.drop(['file'], axis=1)
    pred = pred.drop(['file'], axis=1)

    if mixing_info and mix_df_test is not None:
        average_column = [round(mean_squared_error(target, pred), rounding), None, None, None,
                          round(r2_score(target, pred), rounding), None, None, None]
        metric_rows = ['MSE', 'MSE_b', 'MSE_w', 'MSE_h', 'R2', 'R2_b', 'R2_w', 'R2_h']

        df_metrics_test = pd.DataFrame({'Total Average': average_column},
                                       columns=['Total Average'], index=metric_rows)
        for t in target_names:
            pred_m, target_m = pred_and_true_mix(mix_df_test, pred[t].values, target[t].values)
            MSE = round(mean_squared_error(target[t], pred[t]), rounding)
            MSE_b = round(mean_squared_error(target_m[0], pred_m[0]), rounding)
            MSE_w = round(mean_squared_error(target_m[1], pred_m[1]), rounding)
            MSE_h = round(mean_squared_error(target_m[2], pred_m[2]), rounding)
            R2 = round(r2_score(target[t], pred[t]), rounding)
            R2_b = round(r2_score(target_m[0], pred_m[0]), rounding)
            R2_w = round(r2_score(target_m[1], pred_m[1]), rounding)
            R2_h = round(r2_score(target_m[2], pred_m[2]), rounding)
            df_metrics_test[t] = [MSE, MSE_b, MSE_w, MSE_h, R2, R2_b, R2_w, R2_h]
        for i in df_metrics_test.index:
            if i not in ['MSE', 'R2:']:
                MSE_avg = round(float(np.mean(df_metrics_test.loc[i, target_names[0]:])), rounding)
                df_metrics_test.at[i, 'Total Average'] = MSE_avg

    elif mixing_info and mix_df_test is None:
        print("Pass mix_df_test if you want mixing_info")

    else:
        average_column = [mean_squared_error(target, pred),
                          mean_absolute_error(target, pred),
                          r2_score(target, pred),
                          hf.adj_r2(target, pred, data)]
        metric_rows = ['MSE', 'MAE', 'R2', 'R2 adjusted']
        df_metrics_test = pd.DataFrame({'Total Average': average_column},
                                       columns=['Total Average'], index=metric_rows)
        for t in target_names:
            MSE = round(mean_squared_error(target[t], pred[t]), rounding)
            MAE = round(mean_absolute_error(target[t], pred[t]), rounding)
            R2 = round(r2_score(target[t], pred[t]), rounding)
            ADJ_R2 = round(hf.adj_r2(target[t], pred[t], data), rounding)
            df_metrics_test[t] = [MSE, MAE, R2, ADJ_R2]

    print("")
    return df_metrics_test


def delete_cp(it):
    folder = 'training_' + str(it)
    if os.path.exists(folder):
        open(folder + '/saved_model.pb', 'w').close()
        shutil.rmtree(dir, ignore_errors=True)
    else:
        pass
        # print("Directory does not exist")
