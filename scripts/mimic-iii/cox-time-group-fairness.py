from time import localtime, strftime

import best_parameters
import cohort.get_cohort as sa_cohort
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import settings
import torch
import torchtuples as tt
from scipy import stats
from pycox import utils
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
from pycox.models.cox_time import MixedInputMLPCoxTime
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn_pandas import DataFrameMapper


def cohort_samples_group_fairness(seed, size, cohort):
    # Feature transforms
    labtrans = CoxTime.label_transform()

    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples

    # Sample size: 1182
    test_dataset_women = test_dataset.loc[test_dataset["gender"] == 1]
    # Sample size: 1719
    test_dataset_men = test_dataset.loc[test_dataset["gender"] == 0]
    # Sample size: 353
    test_dataset_black = test_dataset.loc[test_dataset["ethnicity_grouped"] == "black"]
    # Sample size: 2375
    test_dataset_white = test_dataset.loc[test_dataset["ethnicity_grouped"] == "white"]
    # Sample size: 195
    test_dataset_women_black = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black")]
    # Sample size: 939
    test_dataset_women_white = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white")]
    # Sample size: 158
    test_dataset_men_black = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black")]
    # Sample size: 1436
    test_dataset_men_white = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white")]

    # Preprocess input
    x_train, x_val, x_test_women = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women)
    x_train, x_val, x_test_men = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men)
    x_train, x_val, x_test_black = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black)
    x_train, x_val, x_test_white = preprocess_input_features(train_dataset, valid_dataset, test_dataset_white)

    x_train, x_val, x_test_women_black = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_black)
    x_train, x_val, x_test_women_white = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_white)
    x_train, x_val, x_test_men_black = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_black)
    x_train, x_val, x_test_men_white = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_white)

    # Preprocess target
    train, val, test_women = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women, test_dataset_women, labtrans)
    train, val, test_men = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men, test_dataset_men, labtrans)
    train, val, test_black = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black, test_dataset_black, labtrans)
    train, val, test_white = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_white, test_dataset_white, labtrans)

    train, val, test_women_black = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_black, test_dataset_women_black, labtrans)
    train, val, test_women_white = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_white, test_dataset_women_white, labtrans)
    train, val, test_men_black = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_black, test_dataset_men_black, labtrans)
    train, val, test_men_white = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_white, test_dataset_men_white, labtrans)

    test_datasets = [test_women, test_men, test_black, test_white, test_women_black, test_women_white, test_men_black, test_men_white]
    return train, val, test_datasets, labtrans


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


def cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test, test_dataset, labtrans):
    get_target = lambda df: (df['los_hospital'].values, df['hospital_expire_flag'].values)

    y_train = labtrans.fit_transform(*get_target(train_dataset))
    train = tt.tuplefy(x_train, y_train)

    y_val = labtrans.transform(*get_target(valid_dataset))
    val = tt.tuplefy(x_val, y_val)

    y_test = labtrans.transform(*get_target(test_dataset))
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


def cox_time_fit_and_predict(survival_analysis_model, train, val,
                             lr, batch, dropout, epoch, weight_decay,
                             num_nodes, shrink, device, labtrans):

    net = cox_time_make_net(train, dropout, num_nodes)

    optimizer = tt.optim.AdamWR(decoupled_weight_decay=weight_decay)
    model = survival_analysis_model(net, device=device, optimizer=optimizer, shrink=shrink, labtrans=labtrans)
    model.optimizer.set_lr(lr)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(train[0], train[1], batch, epoch, callbacks, val_data=val.repeat(10).cat())

    _ = model.compute_baseline_hazards()
    return model


def cox_time_survival_function(model, test_datasets_list):
    # Predict survival curve from first group
    surv_women = model.predict_surv_df(test_datasets_list[0][0])
    surv_men = model.predict_surv_df(test_datasets_list[1][0])
    surv_black = model.predict_surv_df(test_datasets_list[2][0])
    surv_white = model.predict_surv_df(test_datasets_list[3][0])

    # Plotting survival curve for first group
    survival_curve_plot(surv_women, surv_men, "women", "men", "women-men")
    survival_curve_plot(surv_black, surv_white, "black", "white", "black-white")

    # Predict survival curve from second group
    surv_women_black = model.predict_surv_df(test_datasets_list[4][0])
    surv_women_white = model.predict_surv_df(test_datasets_list[5][0])
    surv_men_black = model.predict_surv_df(test_datasets_list[6][0])
    surv_men_white = model.predict_surv_df(test_datasets_list[7][0])

    # Plotting survival curve for second group
    survival_curve_plot(surv_women_black, surv_women_white,  "black-women", "white-women", "black-white-women")
    survival_curve_plot(surv_men_black, surv_men_white,  "black-men", "white-men", "black-white-men")


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


def survival_curve_median_calc(surv1, surv2):
    # Median and standard deviation
    df_surv_median1 = surv1.median(axis=1)
    df_surv_std1 = surv1.std(axis=1)
    df_surv_median2 = surv2.median(axis=1)
    df_surv_std2 = surv2.std(axis=1)

    return df_surv_median1, df_surv_std1, df_surv_median2, df_surv_std2


def survival_curve_plot(surv1, surv2, label1, label2, group_name):
    df_surv_median1, df_surv_std1, df_surv_median2, df_surv_std2 = survival_curve_median_calc(surv1, surv2)

    # Compute stats
    test = compute_stats(df_surv_median1, df_surv_median2)

    # 95% Confidence interval
    ci1_left = surv1.quantile(0.025, axis=1)
    ci1_right = surv1.quantile(0.975, axis=1)
    ci2_left = surv2.quantile(0.025, axis=1)
    ci2_right = surv2.quantile(0.975, axis=1)

    # Plot curves
    ax = df_surv_median1.plot(label=label1, color='turquoise', linestyle='--')
    ax.fill_between(df_surv_median1.index, ci1_left, ci1_right, alpha=0.2, facecolor='turquoise')

    ax.plot(df_surv_median2, label=label2, color='slateblue', linestyle='-.')
    ax.fill_between(df_surv_median2.index, ci2_left, ci2_right, alpha=0.2, facecolor='slateblue')

    plt.text(0.5, 0.7, str(test), fontsize=4, transform=plt.gcf().transFigure)
    plt.legend(loc="upper right")
    plt.ylabel('S(t | x)')
    plt.xlabel('Time')

    # Save image
    fig_time = strftime("%d%m%Y%H%M%S", localtime())
    fig_path = "img/cox-time/fairness/group-fairness/cox-time-group-fairness-"
    ax.get_figure().savefig(fig_path + group_name + "-" + fig_time + ".png", format="png", bbox_inches="tight", dpi=600)
    plt.close()


def compute_stats(rvs1, rvs2):
    test_stat = stats.ks_2samp(rvs1, rvs2)
    return test_stat


def main(seed, index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Group fairness
    cohort = sa_cohort.cox_neural_network()
    train, val, test_datasets_list, labtrans = cohort_samples_group_fairness(seed=seed, size=settings.size, cohort=cohort)

    # Neural network
    best = best_parameters.cox_time[index]
    model = cox_time_fit_and_predict(CoxTime, train, val,
                                     lr=best['lr'], batch=best['batch'], dropout=best['dropout'],
                                     epoch=best['epoch'], weight_decay=best['weight_decay'],
                                     num_nodes=best['num_nodes'], shrink=best['shrink'],
                                     device=device, labtrans=labtrans)

    # Plot survival function
    cox_time_survival_function(model, test_datasets_list)


if __name__ == "__main__":
    # second best seed --> 224796801 (n.27)
    main(224796801, 27)
