import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from cohort import get_cohort as gh
from pycox import utils
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
from pycox.models.cox_time import MixedInputMLPCoxTime
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn_pandas import DataFrameMapper


def get_cohort():
    # Get data
    cohort = gh.get_cohort()

    # Binning
    cohort['age_st'] = pd.cut(cohort['age'], np.arange(15, 91, 15))

    # Neural network
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age']
    cohort.drop(drop, axis=1, inplace=True)

    # Gender: from categorical to numerical
    cohort.gender.replace(to_replace=dict(F=1, M=0), inplace=True)
    cohort = cohort.astype({'admission_type': 'category', 'ethnicity_grouped': 'category', 'insurance': 'category',
                            'icd_alzheimer': 'int64', 'icd_cancer': 'int64', 'icd_diabetes': 'int64', 'icd_heart': 'int64',
                            'icd_transplant': 'int64', 'gender': 'int64', 'hospital_expire_flag': 'int64',
                            'oasis_score': 'int64'}, copy=False)
    return cohort


def cohort_samples(seed, cohort):
    _ = torch.manual_seed(seed)

    # Train / valid / test split
    test_dataset = cohort.sample(frac=0.2)
    train_dataset = cohort.drop(test_dataset.index)
    valid_dataset = train_dataset.sample(frac=0.2)
    train_dataset = train_dataset.drop(valid_dataset.index)

    # Feature transforms
    labtrans = CoxTime.label_transform()
    x_train, x_val, x_test = preprocess_input_features(train_dataset, valid_dataset, test_dataset)
    train, val, test = cox_time_preprocess_target_features(x_train, x_val, x_test,
                                                           train_dataset, valid_dataset, test_dataset, labtrans)

    return train, val, test, labtrans


def preprocess_input_features(train_dataset, valid_dataset, test_dataset):
    cols_categorical = ['insurance', 'ethnicity_grouped', 'age_st', 'oasis_score', 'admission_type']
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]
    x_mapper_long = DataFrameMapper(categorical)

    cols_leave = ['gender',
                  'icd_alzheimer', 'icd_cancer', 'icd_diabetes', 'icd_heart', 'icd_transplant',
                  'first_hosp_stay', 'first_icu_stay']
    leave = [(col, None) for col in cols_leave]
    x_mapper_float = DataFrameMapper(leave)

    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df).astype('float32'),
                                            x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df).astype('float32'),
                                        x_mapper_long.transform(df))

    x_train = x_fit_transform(train_dataset)
    x_val = x_transform(valid_dataset)
    x_test = x_transform(test_dataset)

    return x_train, x_val, x_test


def cox_time_preprocess_target_features(x_train, x_val, x_test,
                                        train_dataset, valid_dataset, test_dataset, labtrans):

    get_target = lambda df: (df['los_hospital'].values, df['hospital_expire_flag'].values)

    y_train = labtrans.fit_transform(*get_target(train_dataset))
    y_val = labtrans.transform(*get_target(valid_dataset))
    y_test = labtrans.transform(*get_target(test_dataset))

    train = tt.tuplefy(x_train, y_train)
    val = tt.tuplefy(x_val, y_val)
    test = tt.tuplefy(x_test, y_test)

    return train, val, test


def cox_time_make_net(train, dropout, num_nodes):
    num_embeddings = train[0][1].max(0) + 1
    embedding_dims = num_embeddings // 2

    in_features = train[0][0].shape[1]
    batch_norm = True
    net = MixedInputMLPCoxTime(in_features, num_embeddings, embedding_dims,
                               num_nodes, batch_norm, dropout)

    return net


def cox_time_fit_and_predict(survival_analysis_model, train, val, test,
                             lr, batch, dropout, epoch, weight_decay,
                             num_nodes, shrink, device, labtrans):

    net = cox_time_make_net(train, dropout, num_nodes)

    optimizer = tt.optim.AdamWR(decoupled_weight_decay=weight_decay)
    model = survival_analysis_model(net, device=device, optimizer=optimizer, shrink=shrink, labtrans=labtrans)
    model.optimizer.set_lr(lr)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(train[0], train[1], batch, epoch, callbacks, val_data=val.repeat(10).cat())

    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(test[0])
    surv_v = model.predict_surv_df(val[0])
    return surv, surv_v, model, log


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
    durations = sample[1][0]
    events = sample[1][1]

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


def main():

    ##################################
    # PyCox Library
    # https://github.com/havakv/pycox
    #
    # CoxTime
    #
    #     """The Cox-Time model from [1]. A relative risk model without proportional hazards, trained
    #     with case-control sampling.
    #
    #     References:
    #     [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
    #         Time-to-event prediction with neural networks and Cox regression.
    #         Journal of Machine Learning Research, 20(129):1–30, 2019.
    #         http://jmlr.org/papers/v20/18-424.html
    #     """
    #
    ##################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cohort = get_cohort()
    train, val, test, labtrans = cohort_samples(seed=20, cohort=cohort)

    # Open file
    _file = open("files/cox-time.txt", "a")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    # ---------------------
    # Hyperparameter values
    # ---------------------
    # Layers                           {2, 4}
    # Nodes per layer                  {64, 128, 256, 512}
    # Dropout                          {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
    # Weigh decay                      {0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0}
    # Batch size                       {64, 128, 256, 512, 1024}
    # λ(penalty to the loss function)  {0.1, 0.01, 0.001, 0}
    # Learning Rate                    {0.01, 0.001, 0.0001}
    # Log Durations                    {True, False}

    # Best Parameters: {'batch': 2, 'dropout': 3, 'lr': 1, 'num_nodes': 3, 'shrink': 0, 'weight_decay': 4}

    best = {'lr': 0.001,
            'batch_size': 256,
            'dropout': 0.3,
            'weight_decay': 0.02,
            'num_nodes': [512, 512],
            'shrink': 0.1,
            'epoch': 512}

    surv, surv_v, model, log = cox_time_fit_and_predict(CoxTime, train, val, test,
                                                        lr=best['lr'], batch=best['batch_size'], dropout=best['dropout'],
                                                        epoch=best['epoch'], weight_decay=best['weight_decay'],
                                                        num_nodes=best['num_nodes'], shrink=best['shrink'],
                                                        device=device, labtrans=labtrans)

    model.save_net("files/cox-time-net.pt")

    # Train, Val Loss
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    log.plot().get_figure().savefig("img/cox-time-train-val-loss.png", format="png", bbox_inches="tight")

    # Survival estimates as a dataframe
    estimates = 5
    plt.ylabel('S(t | x)')
    plt.xlabel('Time')
    surv.iloc[:, :estimates].plot().get_figure().savefig("img/cox-time-survival-estimates.png", format="png",
                                                         bbox_inches="tight")

    # Evaluate
    cindex_v, bscore_v, bll_v = evaluate(val, surv_v)
    cindex, bscore, bll = evaluate(test, surv)

    # Best Parameters
    _file.write("Best Parameters: " + str(best) + "\n")

    # Scores
    _file.write("Validation \n" 
                "C-Index: " + str(cindex_v) + "\n" +
                "Brier Score: " + str(bscore_v) + "\n" +
                "Binomial Log-Likelihood: " + str(bll_v) + "\n")
    _file.write("Test \n" 
                "C-Index: " + str(cindex) + "\n" +
                "Brier Score: " + str(bscore) + "\n" +
                "Binomial Log-Likelihood: " + str(bll) + "\n")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()
