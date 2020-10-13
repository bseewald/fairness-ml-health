import glob
import os
from time import localtime, strftime

import best_parameters
import matplotlib.pyplot as plt
import cohort.get_cohort as sa_cohort
import numpy as np
import pandas as pd
import settings
import torch
import torchtuples as tt
from pycox import utils
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn_pandas import DataFrameMapper


def cohort_samples_fairness(seed, size, cohort):
    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples

    # samples size: +-1741 / 1182 (fixed)
    train_dataset_women = train_dataset.loc[train_dataset["gender"] == 1]
    test_dataset_women = test_dataset.loc[test_dataset["gender"] == 1]

    # samples size: +-2110 / 1719 (fixed)
    train_dataset_men = train_dataset.loc[train_dataset["gender"] == 0]
    test_dataset_men = test_dataset.loc[test_dataset["gender"] == 0]

    # samples size: +-351 / 353 (fixed)
    train_dataset_black = train_dataset.loc[train_dataset["ethnicity_grouped"] == "black"]
    test_dataset_black = test_dataset.loc[test_dataset["ethnicity_grouped"] == "black"]

    # samples size: +-3275 / 2375 (fixed)
    train_dataset_white = train_dataset.loc[train_dataset["ethnicity_grouped"] == "white"]
    test_dataset_white = test_dataset.loc[test_dataset["ethnicity_grouped"] == "white"]

    # samples size: +-177 / 195 (fixed)
    train_dataset_women_black = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "black")]
    test_dataset_women_black = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black")]

    # samples size: +-1471 / 939 (fixed)
    train_dataset_women_white = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "white")]
    test_dataset_women_white = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white")]

    # samples size: +-174 / 158 (fixed)
    train_dataset_men_black = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "black")]
    test_dataset_men_black = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black")]

    # samples size: +-1804 / 1436 (fixed)
    train_dataset_men_white = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "white")]
    test_dataset_men_white = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white")]

    # Feature transforms
    labtrans = CoxTime.label_transform()

    # preprocess input
    x_train_women, x_test_women = preprocess_input_features(train_dataset_women, test_dataset_women)
    x_train_men, x_test_men = preprocess_input_features(train_dataset_men, test_dataset_men)
    x_train_black, x_test_black = preprocess_input_features(train_dataset_black, test_dataset_black)
    x_train_white, x_test_white = preprocess_input_features(train_dataset_white, test_dataset_white)

    x_train_women_black, x_test_women_black = preprocess_input_features(train_dataset_women_black, test_dataset_women_black)
    x_train_women_white, x_test_women_white = preprocess_input_features(train_dataset_women_white, test_dataset_women_white)
    x_train_men_black, x_test_men_black = preprocess_input_features(train_dataset_men_black, test_dataset_men_black)
    x_train_men_white, x_test_men_white = preprocess_input_features(train_dataset_men_white, test_dataset_men_white)

    # preprocess target
    train_women, test_women = cox_time_preprocess_target_features(x_train_women, train_dataset_women, x_test_women, test_dataset_women, labtrans)
    train_men, test_men = cox_time_preprocess_target_features(x_train_men, train_dataset_men, x_test_men, test_dataset_men, labtrans)
    train_black, test_black = cox_time_preprocess_target_features(x_train_black, train_dataset_black, x_test_black, test_dataset_black, labtrans)
    train_white, test_white = cox_time_preprocess_target_features(x_train_white, train_dataset_white, x_test_white, test_dataset_white, labtrans)

    train_women_black, test_women_black = cox_time_preprocess_target_features(x_train_women_black, train_dataset_women_black, x_test_women_black, test_dataset_women_black, labtrans)
    train_women_white, test_women_white = cox_time_preprocess_target_features(x_train_women_white, train_dataset_women_white, x_test_women_white, test_dataset_women_white, labtrans)
    train_men_black, test_men_black = cox_time_preprocess_target_features(x_train_men_black, train_dataset_men_black, x_test_men_black, test_dataset_men_black, labtrans)
    train_men_white, test_men_white = cox_time_preprocess_target_features(x_train_men_white, train_dataset_men_white, x_test_men_white, test_dataset_men_white, labtrans)

    return test_women, test_men, test_black, test_white, test_women_black, test_women_white, test_men_black, test_men_white, labtrans


def preprocess_input_features(train_dataset, test_dataset):
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
    x_test = x_transform(test_dataset)

    return x_train, x_test


def cox_time_preprocess_target_features(x_train, train_dataset, x_test, test_dataset, labtrans):
    get_target = lambda df: (df['los_hospital'].values, df['hospital_expire_flag'].values)

    y_train = labtrans.fit_transform(*get_target(train_dataset))
    train = tt.tuplefy(x_train, y_train)

    y_test = labtrans.transform(*get_target(test_dataset))
    test = tt.tuplefy(x_test, y_test)

    return train, test


def cox_time_reload_model(net, weight_decay, shrink, device):
    # Load model
    optimizer = tt.optim.AdamWR(decoupled_weight_decay=weight_decay)
    model = CoxTime(net, device=device, optimizer=optimizer, shrink=shrink)
    return model


def add_km_censor_modified(ev, durations, events):
    """
        Add censoring estimates obtained by Kaplan-Meier on the test set(durations, 1-events).
    """
    # modified add_km_censor function
    km = utils.kaplan_meier(durations, 1 - events)
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


def survival_curve_plot(surv1, surv2, label1, label2, group_name):
    plt.ylabel('S(t | x)')
    plt.xlabel('Time')
    plt.grid(True)

    df_surv_median1 = surv1.median(axis=1)
    df_surv_std1 = surv1.std(axis=1)
    df_surv_median2 = surv2.median(axis=1)
    df_surv_std2 = surv2.std(axis=1)

    ax = df_surv_median1.plot(label=label1, color='turquoise', linestyle='--')
    ax.fill_between(df_surv_median1.index, df_surv_median1 - df_surv_std1, df_surv_median1 + df_surv_std1, alpha=0.5, facecolor='turquoise')
    ax.plot(df_surv_median2, label=label2, color='slateblue', linestyle='-.')
    ax.fill_between(df_surv_median2.index, df_surv_median2 - df_surv_std2, df_surv_median2 + df_surv_std2, alpha=0.5, facecolor='slateblue')
    plt.legend(loc="upper right")

    fig_time = strftime("%d%m%Y%H%M%S", localtime())
    fig_path = "img/cox-time/group-fairness/cox-time-group-fairness-"
    ax.get_figure().savefig(fig_path + group_name + "-" + fig_time + ".png", format="png", bbox_inches="tight", dpi=600)
    plt.close()


def main(seed, file, index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Group fairness
    cohort = sa_cohort.cox_neural_network()
    test_women, test_men, test_black, test_white, test_women_black, test_women_white, test_men_black, test_men_white, labtrans = cohort_samples_fairness(seed=seed, size=settings.size, cohort=cohort)

    # Reload model
    best = best_parameters.cox_time[index]
    model = cox_time_reload_model(file, weight_decay=best['weight_decay'], shrink=best['shrink'], device=device)

    # Predict survival curve from first group
    surv_women = model.predict_surv_df(test_women[0])
    surv_men = model.predict_surv_df(test_men[0])
    surv_black = model.predict_surv_df(test_black[0])
    surv_white = model.predict_surv_df(test_white[0])

    # Plotting survival curve from first group
    survival_curve_plot(surv_women, surv_men, "women", "men", "women-men")
    survival_curve_plot(surv_black, surv_white, "black", "white", "black-white")

    # Predict survival curve from second group
    surv_women_black = model.predict_surv_df(test_women_black[0])
    surv_women_white = model.predict_surv_df(test_women_white[0])
    surv_men_black = model.predict_surv_df(test_men_black[0])
    surv_men_white = model.predict_surv_df(test_men_white[0])

    # Plotting survival curve from second group
    survival_curve_plot(surv_women_black, surv_women_white,  "black-women", "white-women", "black-white-women")
    survival_curve_plot(surv_men_black, surv_men_white,  "black-men", "white-men", "black-white-men")


if __name__ == "__main__":
    # second best seed --> 224796801 (n.27)
    files = sorted(glob.glob(settings.PATH + "*.pt"), key=os.path.getmtime)
    i = 0
    for seed in settings.seed:
        main(seed, files[i], i)
        i+=1
