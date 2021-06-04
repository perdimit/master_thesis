import itertools
import inspect


def metrics_filter(monitored_metrics, metrics_report, optim='model_val_loss'):

    metrics = ['train_elbo', 'train_loss', 'train_negloglik', 'train_rmse', 'kl_elbo', 'kl_loss',
               'train_r2', 'mix_res_stddev', 'mix_res_mean', 'epis_res_stddev', 'epis_res_mean', 'aleo_res_stddev', 'aleo_res_mean', 'cds', 'caos', 'caus', 'stopped_epoch', 'best_epoch', 'nn_time', 'bnn_time', 'pred_time']
    if metrics_report == 'testing':
        metrics = metrics + ['test_elbo', 'test_loss', 'test_negloglik', 'test_rmse', 'test_r2']
        test_metrics = [m for m in monitored_metrics if m in metrics]
        removed = list(
            set(monitored_metrics) - set(test_metrics))
        if len(removed) >= 1:
            print("The following metrics were not available for final output: %s" % removed)
        for i, m in enumerate(test_metrics):
            if m in ['train_elbo', 'train_loss', 'test_elbo', 'test_loss']:
                test_metrics[i] = m + '(scaled)'
        print("Reported metrics: %s" % test_metrics)
        return {key: [] for key in test_metrics},  test_metrics
    else:
        if optim not in monitored_metrics:
            monitored_metrics.append(optim)
        metrics += ['es_val_elbo', 'model_val_elbo', 'es_val_loss', 'model_val_loss', 'es_val_negloglik',
                    'model_val_negloglik', 'es_val_rmse', 'model_val_rmse', 'kl_elbo', 'kl_loss', 'es_val_r2',
                    'model_val_r2']
        metrics_ = [m for m in monitored_metrics if m in metrics]

        # if custom_model_bool:
        in_train = ['train_elbo', 'es_val_elbo', 'model_val_elbo', 'train_loss', 'es_val_loss', 'model_val_loss',
                    'model_val_negloglik', 'kl_loss', 'stopped_epoch', 'best_epoch']
        # else:
        #     in_train = ('train_loss', 'es_val_loss', 'stopped_epoch', 'best_epoch')

        not_in_final = ['train_loss', 'es_val_loss', 'model_val_loss', 'train_elbo', 'es_val_elbo', 'model_val_elbo']

        training_metrics = [m for m in metrics if
                            m in in_train]

        final_metrics = [m for m in metrics_ if m not in not_in_final]
        # print(final_metrics)
        # if monitored_metrics is None or 'model_val_loss' in monitored_metrics and custom_model_bool:
        optim_scaled = optim + '(scaled)'
        # if custom_model_bool:
        final_metrics.insert(0, optim_scaled)

        removed = list(
            set(monitored_metrics + [optim_scaled]) - set(final_metrics + [optim]))
        if len(removed) >= 1:
            print("The following metrics were not available for final output: %s" % removed)
        print("Reported metrics: %s" % final_metrics)
        return {key: [] for key in training_metrics}, {key: [] for key in
                                                       final_metrics}, training_metrics, final_metrics


def distribute_distributions_parameters(param_dict):
    dist_dict = param_dict.get("distributions")
    if dist_dict is not None:
        all_dist_perm = []
        for pair_dict in dist_dict:
            for i, dist in enumerate(pair_dict):
                if dist.get("prior") is not None:
                    dict_prior = dist.copy()
                    prior_object = dict_prior['prior']
                    del dict_prior['prior']
                    keys, values = zip(*dict_prior.items())
                    prior_permutations = [{**{'prior': prior_object}, **dict(zip(keys, v))} for v in
                                          itertools.product(*values)]

                elif dist.get("pmf") is not None:
                    dict_pmf = dist.copy()
                    pmf_object = dict_pmf['pmf']
                    del dict_pmf['pmf']
                    keys, values = zip(*dict_pmf.items())
                    pmf_permutations = [{**{'pmf': pmf_object}, **dict(zip(keys, v))} for v in
                                        itertools.product(*values)]
                else:
                    print("dict_handling.py: Need to have both prior and pmf defined. Check if link_object is properly set in prior/pmf")
                    break
            dist_ls = list(itertools.product(prior_permutations, pmf_permutations))
            all_dist_perm += dist_ls
    del param_dict["distributions"]
    param_dict["distributions"] = all_dist_perm
    return param_dict


def get_distributions_parameters(dictionary, nn_inputs):
    dist_params = dictionary['distributions']
    prior = dist_params[0]['prior']
    pmf = dist_params[1]['pmf']
    prior_params = dist_params[0].copy()
    pmf_params = dist_params[1].copy()
    del prior_params['prior']
    del pmf_params['pmf']
    set_pmf = getattr(pmf, 'set' + pmf.initializer_type.__name__)
    set_prior = getattr(prior, 'set' + prior.initializer_type.__name__)
    pmf_arg_keys = list(inspect.getfullargspec(set_pmf))[0][1:]
    prior_arg_keys = list(inspect.getfullargspec(set_prior))[0][1:]

    if hasattr(prior, '_set_neural_network') and prior.link_object is None:
        for k, v in nn_inputs.items():
            if k not in prior_params:
                prior_params[k] = v
        # prior_params = {**prior_params, **nn_inputs}
    if hasattr(pmf, '_set_neural_network') and pmf.link_object is None:
        for k, v in nn_inputs.items():
            if k not in prior_params:
                prior_params[k] = v
        # prior_params = {**prior_params, **nn_inputs}

    # Setting prior and posterior parameters
    # Fix can't del
    for k, v in prior_params.items():
        if k not in prior_arg_keys:
            print("(%s) is not a parameter of the current model" % k)
            # del prior_params[k]

    for k, v in pmf_params.items():
        if k not in pmf_arg_keys:
            print("(%s) is not a parameter of the current model" % k)
            # del pmf_params[k]

    return prior_params, pmf_params, prior_arg_keys, pmf_arg_keys, set_prior, set_pmf, prior, pmf


def printable_distributions_dict(dictionary, prior_dist, pmf_dist):
    dist = dictionary['distributions']
    prior_str = '_'.join(
        str(dist[0]['prior'].initializer_type).split(' ')[2].split('.')[1][1:].split('_')[:-1])
    pmf_str = '_'.join(str(dist[1]['pmf'].initializer_type).split(' ')[2].split('.')[1][1:].split('_')[:-1])
    prior_dict = {'prior': prior_dist, 'prior_init': prior_str}
    pmf_dict = {'pmf': pmf_dist, 'pmf_init': pmf_str}
    prior_params = {key: dist[0][key] for key in dist[0] if key not in ['prior']}
    pmf_params = {key: dist[1][key] for key in dist[1] if key not in ['pmf']}
    for key in prior_params:
        if key in dictionary:
            prior_params[key + '_nn'] = prior_params.pop(key)
    for key in pmf_params:
        if key in dictionary:
            pmf_params[key + '_nn'] = pmf_params.pop(key)
    new_dist_dict = {**prior_dict, **prior_params, **pmf_dict, **pmf_params}
    return new_dist_dict
