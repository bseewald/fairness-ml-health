from time import localtime, strftime

import best_parameters
import numpy as np
import pandas as pd
import settings
import hyperopt_parameters as parameters
import torch
import torchtuples as tt
from hyperopt import STATUS_OK
from lifelines.datasets import load_rossi
from pycox import utils
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler


def split(seed, size, cohort):
    df_train = cohort.drop('age', axis=1)
    df_test = df_train.sample(frac=size, random_state=seed)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=size, random_state=seed)
    df_train = df_train.drop(df_val.index)
    return df_train, df_val, df_test


def cohort_samples(seed, size, cohort):
    # Feature transforms
    labtrans = CoxTime.label_transform()
    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = split(seed, size, cohort)
    # Preprocess input
    x_train, x_val, x_test = preprocess_input_features(train_dataset, valid_dataset, test_dataset)
    # Preprocess target
    train, val, test = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test, test_dataset, labtrans)

    return train, val, test, train_dataset, valid_dataset, test_dataset, labtrans


def cohort_samples_fairness(train_dataset, valid_dataset, test_dataset, labtrans):

    # Sample size: 379
    df_black = test_dataset.loc[test_dataset["race"] == 1]
    test_dataset_black = df_black.sample(frac=1, replace=True)

    # Sample size: 53
    df_other = test_dataset.loc[test_dataset["race"] == 0]
    test_dataset_other = df_other.sample(frac=1, replace=True)

    # Sample size: 194
    df_black_fin = test_dataset.loc[(test_dataset["race"] == 1) & (test_dataset["fin"] == 1)]
    test_dataset_black_fin = df_black_fin.sample(frac=1, replace=True)

    # Sample size: 22
    df_other_fin = test_dataset.loc[(test_dataset["race"] == 0) & (test_dataset["fin"] == 1)]
    test_dataset_other_fin = df_other_fin.sample(frac=1, replace=True)

    # Sample size: 185
    df_black_not_fin = test_dataset.loc[(test_dataset["race"] == 1) & (test_dataset["fin"] == 0)]
    test_dataset_black_not_fin = df_black_not_fin.sample(frac=1, replace=True)

    # Sample size: 31
    df_other_not_fin = test_dataset.loc[(test_dataset["race"] == 0) & (test_dataset["fin"] == 0)]
    test_dataset_other_not_fin = df_other_not_fin.sample(frac=1, replace=True)

    # Preprocess input
    x_train, x_val, x_test_black = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black)
    x_train, x_val, x_test_other = preprocess_input_features(train_dataset, valid_dataset, test_dataset_other)
    x_train, x_val, x_test_black_fin = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_fin)
    x_train, x_val, x_test_black_not_fin = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_not_fin)
    x_train, x_val, x_test_other_fin = preprocess_input_features(train_dataset, valid_dataset, test_dataset_other_fin)
    x_train, x_val, x_test_other_not_fin = preprocess_input_features(train_dataset, valid_dataset, test_dataset_other_not_fin)

    # Preprocess target
    train, val, test_black = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black, test_dataset_black, labtrans)
    train, val, test_other = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_other, test_dataset_other, labtrans)
    train, val, test_black_fin = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_fin, test_dataset_black_fin, labtrans)
    train, val, test_black_not_fin = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_not_fin, test_dataset_black_not_fin, labtrans)
    train, val, test_other_fin = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_other_fin, test_dataset_other_fin, labtrans)
    train, val, test_other_not_fin = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_other_not_fin, test_dataset_other_not_fin, labtrans)

    test_datasets = [test_black, test_other, test_black_fin, test_black_not_fin, test_other_fin, test_other_not_fin]
    return test_datasets


def preprocess_input_features(train_dataset, valid_dataset, test_dataset):
    cols_standardize = ['prio']
    standardize = [([col], StandardScaler()) for col in cols_standardize]

    cols_leave = ['race', 'fin', 'wexp', 'mar', 'paro']
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)

    x_train = x_mapper.fit_transform(train_dataset).astype('float32')
    x_val = x_mapper.transform(valid_dataset).astype('float32')
    x_test = x_mapper.transform(test_dataset).astype('float32')

    return x_train, x_val, x_test


def cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test, test_dataset, labtrans):
    get_target = lambda df: (df['week'].values, df['arrest'].values)

    y_train = labtrans.fit_transform(*get_target(train_dataset))
    train = tt.tuplefy(x_train, y_train)

    y_val = labtrans.transform(*get_target(valid_dataset))
    val = tt.tuplefy(x_val, y_val)

    y_test = labtrans.transform(*get_target(test_dataset))
    test = tt.tuplefy(x_test, y_test)

    return train, val, test


def cox_time_make_net(train, dropout, num_nodes):
    in_features = train[0].shape[1]
    batch_norm = True
    net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
    return net


def cox_time_fit_and_predict(survival_analysis_model, train, val,
                             lr, batch, dropout, epoch, weight_decay,
                             num_nodes, shrink, device, labtrans):

    net = cox_time_make_net(train, dropout, num_nodes)

    optimizer = tt.optim.AdamWR(decoupled_weight_decay=weight_decay)
    model = survival_analysis_model(net, device=device, optimizer=optimizer, shrink=shrink, labtrans=labtrans)
    model.optimizer.set_lr(lr)

    callbacks = [tt.callbacks.EarlyStopping()]
    model.fit(train[0], train[1], batch, epoch, callbacks, val_data=val.repeat(10).cat())

    _ = model.compute_baseline_hazards()
    return model


def add_km_censor_modified(ev, durations, events):

    """
        Add censoring estimates obtained by Kaplan-Meier on the test set(durations, 1-events).
    """
    # modified add_km_censor function
    km = utils.kaplan_meier(durations, 1-events)
    surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(durations), axis=1), index=km.index)

    # increasing index
    if pd.Series(surv.index).is_monotonic is False:
        surv.drop(0.000000, axis=0, inplace=True)

    return ev.add_censor_est(surv)


def evaluate(sample, surv):

    durations = sample[0]
    events = sample[1]

    ev = EvalSurv(surv, durations, events)

    # Setting 'add_km_censor_modified' means that we estimate
    # the censoring distribution by Kaplan-Meier on the test set.
    _ = add_km_censor_modified(ev, durations, events)

    # c-index
    cindex = ev.concordance_td()

    # brier score
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    _ = ev.brier_score(time_grid)
    bscore = ev.integrated_brier_score(time_grid)

    # binomial log-likelihood
    nbll = ev.integrated_nbll(time_grid)

    return cindex, bscore, nbll


def evaluate_modified_cindex(sample_g1, sample_g2, surv_g1, surv_g2):
    durations_g1 = sample_g1[1][0]
    events_g1 = sample_g1[1][1]

    durations_g2 = sample_g2[1][0]
    events_g2 = sample_g2[1][1]

    ev_g1 = EvalSurv(surv_g1, durations_g1, events_g1)
    ev_g2 = EvalSurv(surv_g2, durations_g2, events_g2)

    # c-index
    cindex_g1 = ev_g1.concordance_td()
    cindex_g2 = ev_g2.concordance_td()

    # modified c-index
    cindex_modified_g1 = ev_g1.concordance_td_modified_for_groups(durations_g2, events_g2, surv_g2)
    cindex_modified_g2 = ev_g2.concordance_td_modified_for_groups(durations_g1, events_g1, surv_g1)

    # modified c-index time event
    cindex_modified_g1_time_event = ev_g1.concordance_td_modified_for_groups_time_event(durations_g2, events_g2, surv_g2)
    cindex_modified_g2_time_event = ev_g2.concordance_td_modified_for_groups_time_event(durations_g1, events_g1, surv_g1)

    return cindex_g1, cindex_g2, cindex_modified_g1, cindex_modified_g2, cindex_modified_g1_time_event, cindex_modified_g2_time_event


def bootstrap_cindex(model, test_datasets_list):
    # Predict survival
    surv_black = model.predict_surv_df(test_datasets_list[0][0])
    surv_other = model.predict_surv_df(test_datasets_list[1][0])
    surv_black_fin = model.predict_surv_df(test_datasets_list[2][0])
    surv_other_fin = model.predict_surv_df(test_datasets_list[3][0])
    surv_black_not_fin = model.predict_surv_df(test_datasets_list[4][0])
    surv_other_not_fin = model.predict_surv_df(test_datasets_list[5][0])

    cindex_black, cindex_other, cindex_modified_black_other, cindex_modified_other_black, cindex_modified_black_other_time_event, cindex_modified_other_black_time_event = evaluate_modified_cindex(test_datasets_list[0], test_datasets_list[1], surv_black, surv_other)
    cindex_black_fin, cindex_other_fin, cindex_modified_black_other_fin, cindex_modified_other_black_fin, cindex_modified_black_other_fin_time_event, cindex_modified_other_black_fin_time_event = evaluate_modified_cindex(test_datasets_list[2], test_datasets_list[3], surv_black_fin, surv_other_fin)
    cindex_black_not_fin, cindex_other_not_fin, cindex_modified_black_other_not_fin, cindex_modified_other_black_not_fin, cindex_modified_black_other_not_fin_time_event, cindex_modified_other_black_not_fin_time_event = evaluate_modified_cindex(test_datasets_list[4], test_datasets_list[5], surv_black_not_fin, surv_other_not_fin)

    cindex_list = [cindex_black, cindex_other, cindex_black_fin, cindex_other_fin, cindex_black_not_fin, cindex_other_not_fin]

    modified_cindex_list = [cindex_modified_black_other, cindex_modified_black_other_fin, cindex_modified_black_other_not_fin,
                            cindex_modified_other_black, cindex_modified_other_black_fin, cindex_modified_other_black_not_fin]

    modified_cindex_time_event_list = [cindex_modified_black_other_time_event, cindex_modified_black_other_fin_time_event, cindex_modified_black_other_not_fin_time_event,
                                       cindex_modified_other_black_time_event, cindex_modified_other_black_fin_time_event, cindex_modified_other_black_not_fin_time_event]

    return cindex_list, modified_cindex_list, modified_cindex_time_event_list


def save_txt(path_file, values):
    _file = open(path_file, "a")
    _file.write("\n" + str(values))
    _file.close()


def main(seed, index):
    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cohort = load_rossi()

    # samples
    train, val, test, train_dataset, valid_dataset, test_dataset, labtrans = cohort_samples(seed=seed, size=settings.size, cohort=cohort)

    # train model
    best = best_parameters.cox_time_rossi[index]
    model = cox_time_fit_and_predict(CoxTime, train, val,
                                     lr=best['lr'], batch=best['batch'], dropout=best['dropout'],
                                     epoch=best['epoch'], weight_decay=best['weight_decay'],
                                     num_nodes=best['num_nodes'], shrink=best['shrink'],
                                     device=device, labtrans=labtrans)
    # c-index all base
    surv = model.predict_surv_df(test[0])
    cindex, bscore, bll = evaluate(test[1], surv)
    print(cindex)

    # bootstrap test dataset
    n = 1
    values, values_modified, values_modified_time_event = [], [], []
    for i in range(n):
        test_datasets_list = cohort_samples_fairness(train_dataset, valid_dataset, test_dataset, labtrans)
        cindex_list, modified_cindex_list, modified_cindex_time_event_list = bootstrap_cindex(model, test_datasets_list)
        for j in range(len(cindex_list)):
            values.append(cindex_list[j])
            values_modified.append(modified_cindex_list[j])
            values_modified_time_event.append(modified_cindex_time_event_list[j])

    print(values, values_modified, values_modified_time_event)
    save_txt("files/cox-time/cindex/modified/rossi/black_other.txt", values_modified[0])
    save_txt("files/cox-time/cindex/modified/rossi/black_other_fin.txt", values_modified[1])
    save_txt("files/cox-time/cindex/modified/rossi/black_other_not_fin.txt", values_modified[2])
    save_txt("files/cox-time/cindex/modified/rossi/other_black.txt", values_modified[3])
    save_txt("files/cox-time/cindex/modified/rossi/other_black_fin.txt", values_modified[4])
    save_txt("files/cox-time/cindex/modified/rossi/other_black_not_fin.txt", values_modified[5])

    save_txt("files/cox-time/cindex/modified/rossi/black_other_time_event.txt", values_modified_time_event[0])
    save_txt("files/cox-time/cindex/modified/rossi/black_other_fin_time_event.txt", values_modified_time_event[1])
    save_txt("files/cox-time/cindex/modified/rossi/black_other_not_fin_time_event.txt", values_modified_time_event[2])
    save_txt("files/cox-time/cindex/modified/rossi/other_black_time_event.txt", values_modified_time_event[3])
    save_txt("files/cox-time/cindex/modified/rossi/other_black_fin_time_event.txt", values_modified_time_event[4])
    save_txt("files/cox-time/cindex/modified/rossi/other_black_not_fin_time_event.txt", values_modified_time_event[5])

    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))


def experiment(params):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cohort = load_rossi()
    train, val, test, train_dataset, valid_dataset, test_dataset, labtrans = cohort_samples(seed=params['seed'], size=settings.size, cohort=cohort)

    net = cox_time_make_net(train, params['dropout'], params['num_nodes'])
    optimizer = tt.optim.AdamWR(decoupled_weight_decay=params['weight_decay'])
    model = CoxTime(net, device=device, optimizer=optimizer, shrink=params['shrink'], labtrans=labtrans)
    model.optimizer.set_lr(params['lr'])
    callbacks = [tt.callbacks.EarlyStopping()]
    _ = model.fit(train[0], train[1], batch_size=params['batch'], epochs=settings.epochs, callbacks=callbacks, val_data=val.repeat(10).cat())
    _ = model.compute_baseline_hazards()

    surv = model.predict_surv_df(test[0])
    durations = test[1][0]
    events = test[1][1]
    ev = EvalSurv(surv, durations, events)

    # Setting 'add_km_censor_modified' means that we estimate
    # the censoring distribution by Kaplan-Meier on the test set.
    _ = add_km_censor_modified(ev, durations, events)

    # c-index: the bigger, the better
    cindex = ev.concordance_td()

    # -cindex it's a work around because hyperopt uses fmin function
    return {'loss': -cindex, 'status': STATUS_OK}


def cox_time_optimize(seed):
    # Open file
    _file = open("files/cox-time/cox-time-hyperopt-rossi.txt", "a")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    trials, best = parameters.hyperopt(experiment, seed, "cc")

    # All parameters
    _file.write("All Parameters: \n" + str(trials.trials) + "\n\n")

    # Best Parameters
    _file.write("Best Parameters: " + str(best) + "\n")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    # second best seed
    main(224796801, 0)
    # cox_time_optimize(224796801)
