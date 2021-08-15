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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

OASIS_SCORE = 3


def cohort_samples_fairness_gender(seed, size, cohort):
    # Feature transforms
    labtrans = CoxTime.label_transform()

    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples + legitimate factor (oasis_score, disease)

    # sample size: 254
    test_dataset_women_oasis = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["oasis_score"] == OASIS_SCORE)]
    # sample size: 14
    test_dataset_women_alz = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_alzheimer"] == 1)]
    # sample size: 570
    test_dataset_women_cancer = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_cancer"] == 1)]
    # sample size: 694
    test_dataset_women_diab = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_diabetes"] == 1)]
    # sample size: 640
    test_dataset_women_heart = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_heart"] == 1)]
    # sample size: 108
    test_dataset_women_transp = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_transplant"] == 1)]
    # sample size: 313
    test_dataset_men_oasis = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["oasis_score"] == 3)]
    # sample size: 5
    test_dataset_men_alz = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_alzheimer"] == 1)]
    # sample size: 804
    test_dataset_men_cancer = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_cancer"] == 1)]
    # sample size: 1039
    test_dataset_men_diab = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_diabetes"] == 1)]
    # sample size: 974
    test_dataset_men_heart = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_heart"] == 1)]
    # sample size: 194
    test_dataset_men_transp = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_transplant"] == 1)]

    # Preprocess Input
    x_train, x_val, x_test_women_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_oasis)
    x_train, x_val, x_test_men_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_oasis)

    x_train, x_val, x_test_women_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_alz)
    x_train, x_val, x_test_men_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_alz)

    x_train, x_val, x_test_women_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_cancer)
    x_train, x_val, x_test_men_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_cancer)

    x_train, x_val, x_test_women_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_diab)
    x_train, x_val, x_test_men_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_diab)

    x_train, x_val, x_test_women_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_heart)
    x_train, x_val, x_test_men_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_heart)

    x_train, x_val, x_test_women_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_transp)
    x_train, x_val, x_test_men_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_transp)

    # Preprocess Target
    train, val, test_women_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_oasis, test_dataset_women_oasis, labtrans)
    train, val, test_men_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_oasis, test_dataset_men_oasis, labtrans)

    train, val, test_women_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_alz, test_dataset_women_alz, labtrans)
    train, val, test_men_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_alz, test_dataset_men_alz, labtrans)

    train, val, test_women_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_cancer, test_dataset_women_cancer, labtrans)
    train, val, test_men_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_cancer, test_dataset_men_cancer, labtrans)

    train, val, test_women_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_diab, test_dataset_women_diab, labtrans)
    train, val, test_men_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_diab, test_dataset_men_diab, labtrans)

    train, val, test_women_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_heart, test_dataset_women_heart, labtrans)
    train, val, test_men_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_heart, test_dataset_men_heart, labtrans)

    train, val, test_women_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_transp, test_dataset_women_transp, labtrans)
    train, val, test_men_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_transp, test_dataset_men_transp, labtrans)

    test_datasets = [test_women_oasis, test_men_oasis, test_women_alz, test_men_alz, test_women_cancer, test_men_cancer,
                     test_women_diab, test_men_diab, test_women_heart, test_men_heart, test_women_transp, test_men_transp]
    return train, val, test_datasets, labtrans


def cohort_samples_fairness_race(seed, size, cohort):
    # Feature transforms
    labtrans = CoxTime.label_transform()

    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples + legitimate factor (oasis_score, disease)
    # other possibilities: admission_type, insurance, age_st

    # sample size: 81
    test_dataset_black_oasis = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["oasis_score"] == OASIS_SCORE)]
    # sample size: 6
    test_dataset_black_alz = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_alzheimer"] == 1)]
    # sample size: 116
    test_dataset_black_cancer = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_cancer"] == 1)]
    # sample size: 267
    test_dataset_black_diab = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_diabetes"] == 1)]
    # sample size: 183
    test_dataset_black_heart = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_heart"] == 1)]
    # sample size: 31
    test_dataset_black_transp = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_transplant"] == 1)]
    # sample size: 456
    test_dataset_white_oasis = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["oasis_score"] == 3)]
    # sample size: 13
    test_dataset_white_alz = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_alzheimer"] == 1)]
    # sample size: 1170
    test_dataset_white_cancer = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_cancer"] == 1)]
    # sample size: 1371
    test_dataset_white_diab = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_diabetes"] == 1)]
    # sample size: 1358
    test_dataset_white_heart = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_heart"] == 1)]
    # sample size: 255
    test_dataset_white_transp = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_transplant"] == 1)]

    # Preprocess input
    x_train, x_val, x_test_black_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_oasis)
    x_train, x_val, x_test_white_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_white_oasis)

    x_train, x_val, x_test_black_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_alz)
    x_train, x_val, x_test_white_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_white_alz)

    x_train, x_val, x_test_black_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_cancer)
    x_train, x_val, x_test_white_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_white_cancer)

    x_train, x_val, x_test_black_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_diab)
    x_train, x_val, x_test_white_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_white_diab)

    x_train, x_val, x_test_black_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_heart)
    x_train, x_val, x_test_white_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_white_heart)

    x_train, x_val, x_test_black_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_black_transp)
    x_train, x_val, x_test_white_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_white_transp)

    # Preprocess target
    train, val, test_black_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_oasis, test_dataset_black_oasis, labtrans)
    train, val, test_white_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_white_oasis, test_dataset_white_oasis, labtrans)

    train, val, test_black_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_alz, test_dataset_black_alz, labtrans)
    train, val, test_white_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_white_alz, test_dataset_white_alz, labtrans)

    train, val, test_black_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_cancer, test_dataset_black_cancer, labtrans)
    train, val, test_white_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_white_cancer, test_dataset_white_cancer, labtrans)

    train, val, test_black_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_diab, test_dataset_black_diab, labtrans)
    train, val, test_white_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_white_diab, test_dataset_white_diab, labtrans)

    train, val, test_black_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_heart, test_dataset_black_heart, labtrans)
    train, val, test_white_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_white_heart, test_dataset_white_heart, labtrans)

    train, val, test_black_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_black_transp, test_dataset_black_transp, labtrans)
    train, val, test_white_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_white_transp, test_dataset_white_transp, labtrans)

    test_datasets = [test_black_oasis, test_white_oasis, test_black_alz, test_white_alz, test_black_cancer, test_white_cancer,
                     test_black_diab, test_white_diab, test_black_heart, test_white_heart, test_black_transp, test_white_transp]
    return train, val, test_datasets, labtrans


def cohort_samples_fairness_gender_race(seed, size, cohort):
    # Feature transforms
    labtrans = CoxTime.label_transform()

    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples + legitimate factor (oasis_score, disease)
    # other possibilities: admission_type, insurance, age_st

    # Sample size: 47
    test_dataset_women_black_oasis = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["oasis_score"] == OASIS_SCORE)]
    # Sample size: 5
    test_dataset_women_black_alz = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_alzheimer"] == 1)]
    # Sample size: 55
    test_dataset_women_black_cancer = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_cancer"] == 1)]
    # Sample size: 153
    test_dataset_women_black_diab = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_diabetes"] == 1)]
    # Sample size: 113
    test_dataset_women_black_heart = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_heart"] == 1)]
    # Sample size: 10
    test_dataset_women_black_transp = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_transplant"] == 1)]

    # Sample size: 199
    test_dataset_women_white_oasis = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["oasis_score"] == OASIS_SCORE)]
    # Sample size: 9
    test_dataset_women_white_alz = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_alzheimer"] == 1)]
    # Sample size: 487
    test_dataset_women_white_cancer = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_cancer"] == 1)]
    # Sample size: 517
    test_dataset_women_white_diab = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_diabetes"] == 1)]
    # Sample size: 503
    test_dataset_women_white_heart = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_heart"] == 1)]
    # Sample size: 92
    test_dataset_women_white_transp = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_transplant"] == 1)]

    # Sample size: 34
    test_dataset_men_black_oasis = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["oasis_score"] == OASIS_SCORE)]
    # Sample size: 1
    test_dataset_men_black_alz = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_alzheimer"] == 1)]
    # Sample size: 61
    test_dataset_men_black_cancer = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_cancer"] == 1)]
    # Sample size: 114
    test_dataset_men_black_diab = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_diabetes"] == 1)]
    # Sample size: 70
    test_dataset_men_black_heart = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_heart"] == 1)]
    # Sample size: 21
    test_dataset_men_black_transp = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_transplant"] == 1)]

    # Sample size: 257
    test_dataset_men_white_oasis = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["oasis_score"] == 3)]
    # Sample size: 4
    test_dataset_men_white_alz = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_alzheimer"] == 1)]
    # Sample size: 683
    test_dataset_men_white_cancer = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_cancer"] == 1)]
    # Sample size: 854
    test_dataset_men_white_diab = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_diabetes"] == 1)]
    # Sample size: 855
    test_dataset_men_white_heart = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_heart"] == 1)]
    # Sample size: 163
    test_dataset_men_white_transp = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_transplant"] == 1)]

    # Preprocess input
    x_train, x_val, x_test_women_black_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_black_oasis)
    x_train, x_val, x_test_women_white_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_white_oasis)

    x_train, x_val, x_test_women_black_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_black_alz)
    x_train, x_val, x_test_women_white_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_white_alz)

    x_train, x_val, x_test_women_black_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_black_cancer)
    x_train, x_val, x_test_women_white_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_white_cancer)

    x_train, x_val, x_test_women_black_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_black_diab)
    x_train, x_val, x_test_women_white_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_white_diab)

    x_train, x_val, x_test_women_black_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_black_heart)
    x_train, x_val, x_test_women_white_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_white_heart)

    x_train, x_val, x_test_women_black_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_black_transp)
    x_train, x_val, x_test_women_white_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_women_white_transp)

    x_train, x_val, x_test_men_black_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_black_oasis)
    x_train, x_val, x_test_men_white_oasis = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_white_oasis)

    x_train, x_val, x_test_men_black_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_black_alz)
    x_train, x_val, x_test_men_white_alz = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_white_alz)

    x_train, x_val, x_test_men_black_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_black_cancer)
    x_train, x_val, x_test_men_white_cancer = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_white_cancer)

    x_train, x_val, x_test_men_black_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_black_diab)
    x_train, x_val, x_test_men_white_diab = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_white_diab)

    x_train, x_val, x_test_men_black_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_black_heart)
    x_train, x_val, x_test_men_white_heart = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_white_heart)

    x_train, x_val, x_test_men_black_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_black_transp)
    x_train, x_val, x_test_men_white_transp = preprocess_input_features(train_dataset, valid_dataset, test_dataset_men_white_transp)

    # Preprocess target
    train, val, test_women_black_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_black_oasis, test_dataset_women_black_oasis, labtrans)
    train, val, test_women_white_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_white_oasis, test_dataset_women_white_oasis, labtrans)

    train, val, test_women_black_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_black_alz, test_dataset_women_black_alz, labtrans)
    train, val, test_women_white_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_white_alz, test_dataset_women_white_alz, labtrans)

    train, val, test_women_black_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_black_cancer, test_dataset_women_black_cancer, labtrans)
    train, val, test_women_white_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_white_cancer, test_dataset_women_white_cancer, labtrans)

    train, val, test_women_black_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_black_diab, test_dataset_women_black_diab, labtrans)
    train, val, test_women_white_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_white_diab, test_dataset_women_white_diab, labtrans)

    train, val, test_women_black_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_black_heart, test_dataset_women_black_heart, labtrans)
    train, val, test_women_white_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_white_heart, test_dataset_women_white_heart, labtrans)

    train, val, test_women_black_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_black_transp, test_dataset_women_black_transp, labtrans)
    train, val, test_women_white_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_women_white_transp, test_dataset_women_white_transp, labtrans)

    train, val, test_men_black_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_black_oasis, test_dataset_men_black_oasis, labtrans)
    train, val, test_men_white_oasis = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_white_oasis, test_dataset_men_white_oasis, labtrans)

    train, val, test_men_black_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_black_alz, test_dataset_men_black_alz, labtrans)
    train, val, test_men_white_alz = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_white_alz, test_dataset_men_white_alz, labtrans)

    train, val, test_men_black_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_black_cancer, test_dataset_men_black_cancer, labtrans)
    train, val, test_men_white_cancer = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_white_cancer, test_dataset_men_white_cancer, labtrans)

    train, val, test_men_black_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_black_diab, test_dataset_men_black_diab, labtrans)
    train, val, test_men_white_diab = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_white_diab, test_dataset_men_white_diab, labtrans)

    train, val, test_men_black_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_black_heart, test_dataset_men_black_heart, labtrans)
    train, val, test_men_white_heart = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_white_heart, test_dataset_men_white_heart, labtrans)

    train, val, test_men_black_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_black_transp, test_dataset_men_black_transp, labtrans)
    train, val, test_men_white_transp = cox_time_preprocess_target_features(x_train, train_dataset, x_val, valid_dataset, x_test_men_white_transp, test_dataset_men_white_transp, labtrans)

    test_datasets = [test_women_black_oasis, test_women_white_oasis, test_women_black_alz, test_women_white_alz, test_women_black_cancer, test_women_white_cancer, \
                     test_women_black_diab, test_women_white_diab, test_women_black_heart, test_women_white_heart, test_women_black_transp, test_women_white_transp, \
                     test_men_black_oasis, test_men_white_oasis, test_men_black_alz, test_men_white_alz, test_men_black_cancer, test_men_white_cancer, \
                     test_men_black_diab, test_men_white_diab, test_men_black_heart, test_men_white_heart, test_men_black_transp, test_men_white_transp]
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


# def cox_time_reload_model(net, weight_decay, shrink, device, labtrans):
#     # Load model
#     optimizer = tt.optim.AdamWR(decoupled_weight_decay=weight_decay)
#     model = CoxTime(net, device=device, optimizer=optimizer, shrink=shrink, labtrans=labtrans)
#     return model


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
    fig_path = "img/cox-time/fairness/conditional-statistical-parity/cox-time-conditional-statistical-parity-"
    ax.get_figure().savefig(fig_path + group_name + "-" + fig_time + ".png", format="png", bbox_inches="tight", dpi=600)
    plt.close()


def compute_stats(rvs1, rvs2):
    test_stat = stats.ks_2samp(rvs1, rvs2)
    return test_stat


def cox_time_survival_function(model, test_datasets_list_gender, test_datasets_list_race, test_datasets_list_gender_race):
    # Predict survival curve from first group
    surv_women_oasis = model.predict_surv_df(test_datasets_list_gender[0][0])
    surv_men_oasis = model.predict_surv_df(test_datasets_list_gender[1][0])
    surv_women_alz = model.predict_surv_df(test_datasets_list_gender[2][0])
    surv_men_alz = model.predict_surv_df(test_datasets_list_gender[3][0])
    surv_women_cancer = model.predict_surv_df(test_datasets_list_gender[4][0])
    surv_men_cancer = model.predict_surv_df(test_datasets_list_gender[5][0])
    surv_women_diab = model.predict_surv_df(test_datasets_list_gender[6][0])
    surv_men_diab = model.predict_surv_df(test_datasets_list_gender[7][0])
    surv_women_heart = model.predict_surv_df(test_datasets_list_gender[8][0])
    surv_men_heart = model.predict_surv_df(test_datasets_list_gender[9][0])
    surv_women_transp = model.predict_surv_df(test_datasets_list_gender[10][0])
    surv_men_transp = model.predict_surv_df(test_datasets_list_gender[11][0])

    # Plotting survival curve for first group
    survival_curve_plot(surv_women_oasis, surv_men_oasis, "women | oasis score", "men | oasis score", "women-men-oasis")
    survival_curve_plot(surv_women_alz, surv_men_alz, "women | alzheimer", "men | alzheimer", "women-men-alzheimer")
    survival_curve_plot(surv_women_cancer, surv_men_cancer, "women | cancer", "men | cancer", "women-men-cancer")
    survival_curve_plot(surv_women_diab, surv_men_diab, "women | diabetes", "men | diabetes", "women-men-diabetes")
    survival_curve_plot(surv_women_heart, surv_men_heart, "women | heart", "men | heart", "women-men-heart")
    survival_curve_plot(surv_women_transp, surv_men_transp, "women | transplant", "men | transplant", "women-men-transplant")

    # Predict survival curve from second group
    surv_black_oasis = model.predict_surv_df(test_datasets_list_race[0][0])
    surv_white_oasis = model.predict_surv_df(test_datasets_list_race[1][0])
    surv_black_alz = model.predict_surv_df(test_datasets_list_race[2][0])
    surv_white_alz = model.predict_surv_df(test_datasets_list_race[3][0])
    surv_black_cancer = model.predict_surv_df(test_datasets_list_race[4][0])
    surv_white_cancer = model.predict_surv_df(test_datasets_list_race[5][0])
    surv_black_diab = model.predict_surv_df(test_datasets_list_race[6][0])
    surv_white_diab = model.predict_surv_df(test_datasets_list_race[7][0])
    surv_black_heart = model.predict_surv_df(test_datasets_list_race[8][0])
    surv_white_heart = model.predict_surv_df(test_datasets_list_race[9][0])
    surv_black_transp = model.predict_surv_df(test_datasets_list_race[10][0])
    surv_white_transp = model.predict_surv_df(test_datasets_list_race[11][0])

    # Plotting survival curve for second group
    survival_curve_plot(surv_black_oasis, surv_white_oasis, "black | oasis score", "white | oasis score", "black-white-oasis")
    survival_curve_plot(surv_black_alz, surv_white_alz, "black | alzheimer", "white | alzheimer", "black-white-alzheimer")
    survival_curve_plot(surv_black_cancer, surv_white_cancer, "black | cancer", "white | cancer", "black-white-cancer")
    survival_curve_plot(surv_black_diab, surv_white_diab, "black | diabetes", "white | diabetes", "black-white-diabetes")
    survival_curve_plot(surv_black_heart, surv_white_heart, "black | heart", "white | heart", "black-white-heart")
    survival_curve_plot(surv_black_transp, surv_white_transp, "black | transplant", "white | transplant", "black-white-transplant")

    # Predict survival curve from third group
    surv_women_black_oasis = model.predict_surv_df(test_datasets_list_gender_race[0][0])
    surv_women_white_oasis = model.predict_surv_df(test_datasets_list_gender_race[1][0])
    surv_women_black_alz = model.predict_surv_df(test_datasets_list_gender_race[2][0])
    surv_women_white_alz = model.predict_surv_df(test_datasets_list_gender_race[3][0])
    surv_women_black_cancer = model.predict_surv_df(test_datasets_list_gender_race[4][0])
    surv_women_white_cancer = model.predict_surv_df(test_datasets_list_gender_race[5][0])
    surv_women_black_diab = model.predict_surv_df(test_datasets_list_gender_race[6][0])
    surv_women_white_diab = model.predict_surv_df(test_datasets_list_gender_race[7][0])
    surv_women_black_heart = model.predict_surv_df(test_datasets_list_gender_race[8][0])
    surv_women_white_heart = model.predict_surv_df(test_datasets_list_gender_race[9][0])
    surv_women_black_transp = model.predict_surv_df(test_datasets_list_gender_race[10][0])
    surv_women_white_transp = model.predict_surv_df(test_datasets_list_gender_race[11][0])

    # Plotting survival curve for 3.1 group
    survival_curve_plot(surv_women_black_oasis, surv_women_white_oasis, "black women | oasis score", "white women | oasis score", "black-women-white-women-oasis")
    survival_curve_plot(surv_women_black_alz, surv_women_white_alz, "black women | alzheimer", "white women | alzheimer", "black-women-white-women-alzheimer")
    survival_curve_plot(surv_women_black_cancer, surv_women_white_cancer, "black women | cancer", "white women | cancer", "black-women-white-women-cancer")
    survival_curve_plot(surv_women_black_diab, surv_women_white_diab, "black women | diabetes", "white women | diabetes", "black-women-white-women-diabetes")
    survival_curve_plot(surv_women_black_heart, surv_women_white_heart, "black women | heart", "white women | heart", "black-women-white-women-heart")
    survival_curve_plot(surv_women_black_transp, surv_women_white_transp, "black women | transplant", "white women | transplant", "black-women-white-women-transplant")

    # Predict survival curve from 3.2 group
    surv_men_black_oasis = model.predict_surv_df(test_datasets_list_gender_race[0][0])
    surv_men_white_oasis = model.predict_surv_df(test_datasets_list_gender_race[1][0])
    surv_men_black_alz = model.predict_surv_df(test_datasets_list_gender_race[2][0])
    surv_men_white_alz = model.predict_surv_df(test_datasets_list_gender_race[3][0])
    surv_men_black_cancer = model.predict_surv_df(test_datasets_list_gender_race[4][0])
    surv_men_white_cancer = model.predict_surv_df(test_datasets_list_gender_race[5][0])
    surv_men_black_diab = model.predict_surv_df(test_datasets_list_gender_race[6][0])
    surv_men_white_diab = model.predict_surv_df(test_datasets_list_gender_race[7][0])
    surv_men_black_heart = model.predict_surv_df(test_datasets_list_gender_race[8][0])
    surv_men_white_heart = model.predict_surv_df(test_datasets_list_gender_race[9][0])
    surv_men_black_transp = model.predict_surv_df(test_datasets_list_gender_race[10][0])
    surv_men_white_transp = model.predict_surv_df(test_datasets_list_gender_race[11][0])

    # Plotting survival curve from 3.2 group
    survival_curve_plot(surv_men_black_oasis, surv_men_white_oasis, "black men | oasis score", "white men | oasis score", "black-men-white-men-oasis")
    survival_curve_plot(surv_men_black_alz, surv_men_white_alz, "black men | alzheimer", "white men | alzheimer", "black-men-white-men-alzheimer")
    survival_curve_plot(surv_men_black_cancer, surv_men_white_cancer, "black men | cancer", "white men | cancer", "black-men-white-men-cancer")
    survival_curve_plot(surv_men_black_diab, surv_men_white_diab, "black men | diabetes", "white men | diabetes", "black-men-white-men-diabetes")
    survival_curve_plot(surv_men_black_heart, surv_men_white_heart, "black men | heart", "white men | heart", "black-men-white-men-heart")
    survival_curve_plot(surv_men_black_transp, surv_men_white_transp, "black men | transplant", "white men | transplant", "black-men-white-men-transplant")


def cohort_samples_gender(dataset):

    dataset_women_oasis = dataset.loc[(dataset["gender"] == 1) & (dataset["oasis_score"] == OASIS_SCORE)]
    dataset_women_alz = dataset.loc[(dataset["gender"] == 1) & (dataset["icd_alzheimer"] == 1)]
    dataset_women_cancer = dataset.loc[(dataset["gender"] == 1) & (dataset["icd_cancer"] == 1)]
    dataset_women_diab = dataset.loc[(dataset["gender"] == 1) & (dataset["icd_diabetes"] == 1)]
    dataset_women_heart = dataset.loc[(dataset["gender"] == 1) & (dataset["icd_heart"] == 1)]
    dataset_women_transp = dataset.loc[(dataset["gender"] == 1) & (dataset["icd_transplant"] == 1)]

    dataset_men_oasis = dataset.loc[(dataset["gender"] == 0) & (dataset["oasis_score"] == 3)]
    dataset_men_alz = dataset.loc[(dataset["gender"] == 0) & (dataset["icd_alzheimer"] == 1)]
    dataset_men_cancer = dataset.loc[(dataset["gender"] == 0) & (dataset["icd_cancer"] == 1)]
    dataset_men_diab = dataset.loc[(dataset["gender"] == 0) & (dataset["icd_diabetes"] == 1)]
    dataset_men_heart = dataset.loc[(dataset["gender"] == 0) & (dataset["icd_heart"] == 1)]
    dataset_men_transp = dataset.loc[(dataset["gender"] == 0) & (dataset["icd_transplant"] == 1)]

    dataset_list = [dataset_women_oasis, dataset_men_oasis, dataset_women_alz, dataset_men_alz, dataset_women_cancer, dataset_men_cancer,
                    dataset_women_diab, dataset_men_diab, dataset_women_heart, dataset_men_heart, dataset_women_transp, dataset_men_transp]
    return dataset_list


def cohort_samples_race(dataset):

    dataset_black_oasis = dataset.loc[(dataset["ethnicity_grouped"] == "black") & (dataset["oasis_score"] == OASIS_SCORE)]
    dataset_black_alz = dataset.loc[(dataset["ethnicity_grouped"] == "black") & (dataset["icd_alzheimer"] == 1)]
    dataset_black_cancer = dataset.loc[(dataset["ethnicity_grouped"] == "black") & (dataset["icd_cancer"] == 1)]
    dataset_black_diab = dataset.loc[(dataset["ethnicity_grouped"] == "black") & (dataset["icd_diabetes"] == 1)]
    dataset_black_heart = dataset.loc[(dataset["ethnicity_grouped"] == "black") & (dataset["icd_heart"] == 1)]
    dataset_black_transp = dataset.loc[(dataset["ethnicity_grouped"] == "black") & (dataset["icd_transplant"] == 1)]

    dataset_white_oasis = dataset.loc[(dataset["ethnicity_grouped"] == "white") & (dataset["oasis_score"] == 3)]
    dataset_white_alz = dataset.loc[(dataset["ethnicity_grouped"] == "white") & (dataset["icd_alzheimer"] == 1)]
    dataset_white_cancer = dataset.loc[(dataset["ethnicity_grouped"] == "white") & (dataset["icd_cancer"] == 1)]
    dataset_white_diab = dataset.loc[(dataset["ethnicity_grouped"] == "white") & (dataset["icd_diabetes"] == 1)]
    dataset_white_heart = dataset.loc[(dataset["ethnicity_grouped"] == "white") & (dataset["icd_heart"] == 1)]
    dataset_white_transp = dataset.loc[(dataset["ethnicity_grouped"] == "white") & (dataset["icd_transplant"] == 1)]

    dataset_list = [dataset_black_oasis, dataset_white_oasis, dataset_black_alz, dataset_white_alz, dataset_black_cancer, dataset_white_cancer,
                    dataset_black_diab, dataset_white_diab, dataset_black_heart, dataset_white_heart, dataset_black_transp, dataset_white_transp]
    return dataset_list


def cohort_samples_gender_race(dataset):

    dataset_women_black_oasis = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "black") & (dataset["oasis_score"] == OASIS_SCORE)]
    dataset_women_black_alz = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_alzheimer"] == 1)]
    dataset_women_black_cancer = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_cancer"] == 1)]
    dataset_women_black_diab = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_diabetes"] == 1)]
    dataset_women_black_heart = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_heart"] == 1)]
    dataset_women_black_transp = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_transplant"] == 1)]

    dataset_women_white_oasis = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "white") & (dataset["oasis_score"] == OASIS_SCORE)]
    dataset_women_white_alz = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_alzheimer"] == 1)]
    dataset_women_white_cancer = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_cancer"] == 1)]
    dataset_women_white_diab = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_diabetes"] == 1)]
    dataset_women_white_heart = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_heart"] == 1)]
    dataset_women_white_transp = dataset.loc[(dataset["gender"] == 1) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_transplant"] == 1)]

    dataset_men_black_oasis = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "black") & (dataset["oasis_score"] == OASIS_SCORE)]
    dataset_men_black_alz = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_alzheimer"] == 1)]
    dataset_men_black_cancer = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_cancer"] == 1)]
    dataset_men_black_diab = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_diabetes"] == 1)]
    dataset_men_black_heart = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_heart"] == 1)]
    dataset_men_black_transp = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "black") & (dataset["icd_transplant"] == 1)]

    dataset_men_white_oasis = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "white") & (dataset["oasis_score"] == 3)]
    dataset_men_white_alz = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_alzheimer"] == 1)]
    dataset_men_white_cancer = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_cancer"] == 1)]
    dataset_men_white_diab = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_diabetes"] == 1)]
    dataset_men_white_heart = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_heart"] == 1)]
    dataset_men_white_transp = dataset.loc[(dataset["gender"] == 0) & (dataset["ethnicity_grouped"] == "white") & (dataset["icd_transplant"] == 1)]

    dataset_list_women = [dataset_women_black_oasis, dataset_women_white_oasis, dataset_women_black_alz, dataset_women_white_alz, dataset_women_black_cancer, dataset_women_white_cancer, \
                          dataset_women_black_diab, dataset_women_white_diab, dataset_women_black_heart, dataset_women_white_heart, dataset_women_black_transp, dataset_women_white_transp]
    dataset_list_men = [dataset_men_black_oasis, dataset_men_white_oasis, dataset_men_black_alz, dataset_men_white_alz, dataset_men_black_cancer, dataset_men_white_cancer, \
                        dataset_men_black_diab, dataset_men_white_diab, dataset_men_black_heart, dataset_men_white_heart, dataset_men_black_transp, dataset_men_white_transp]
    return dataset_list_women, dataset_list_men


def logrank_stats(dataset_list):
    results1 = logrank_test(dataset_list[0].los_hospital, dataset_list[1].los_hospital,
                            event_observed_A=dataset_list[0].hospital_expire_flag, event_observed_B=dataset_list[1].hospital_expire_flag)

    results2 = logrank_test(dataset_list[2].los_hospital, dataset_list[3].los_hospital,
                            event_observed_A=dataset_list[2].hospital_expire_flag, event_observed_B=dataset_list[3].hospital_expire_flag)

    results3 = logrank_test(dataset_list[4].los_hospital, dataset_list[5].los_hospital,
                            event_observed_A=dataset_list[4].hospital_expire_flag, event_observed_B=dataset_list[5].hospital_expire_flag)

    results4 = logrank_test(dataset_list[6].los_hospital, dataset_list[7].los_hospital,
                            event_observed_A=dataset_list[6].hospital_expire_flag, event_observed_B=dataset_list[7].hospital_expire_flag)

    results5 = logrank_test(dataset_list[8].los_hospital, dataset_list[9].los_hospital,
                            event_observed_A=dataset_list[8].hospital_expire_flag, event_observed_B=dataset_list[9].hospital_expire_flag)

    results6 = logrank_test(dataset_list[10].los_hospital, dataset_list[11].los_hospital,
                            event_observed_A=dataset_list[10].hospital_expire_flag, event_observed_B=dataset_list[11].hospital_expire_flag)

    print("stats: " + str(results1.test_statistic) + " p-value: " + str(results1.p_value))
    print("stats: " + str(results2.test_statistic) + " p-value: " + str(results2.p_value))
    print("stats: " + str(results3.test_statistic) + " p-value: " + str(results3.p_value))
    print("stats: " + str(results4.test_statistic) + " p-value: " + str(results4.p_value))
    print("stats: " + str(results5.test_statistic) + " p-value: " + str(results5.p_value))
    print("stats: " + str(results6.test_statistic) + " p-value: " + str(results6.p_value))


def km(dataset_list_a, dataset_list_b, group_name, label_a, label_b):

    with plt.style.context('ggplot'):
        kmf = KaplanMeierFitter()

        kmf.fit(dataset_list_a['los_hospital'], dataset_list_a['hospital_expire_flag'], label=label_a)
        ax = kmf.plot_survival_function()
        kmf.fit(dataset_list_b['los_hospital'], dataset_list_b['hospital_expire_flag'], label=label_b)
        ax = kmf.plot_survival_function(ax=ax)

        ax.set_ylabel('S(t|w, z)')
        ax.set_xlabel('Tempo')

        # Save image
        fig_time = strftime("%d%m%Y%H%M%S", localtime())
        fig_path = "img/cox-time/fairness/conditional-statistical-parity/cox-time-conditional-statistical-parity-"
        ax.get_figure().savefig(fig_path + group_name + ".png", format="png", bbox_inches="tight", dpi=600)
        plt.close()


def survival_plot_gender(dataset_list):
    km(dataset_list[0], dataset_list[1], "women-men-oasis", "Mulheres | Oasis", "Homens | Oasis")
    km(dataset_list[2], dataset_list[3], "women-men-alzheimer", "Mulheres | Alzheimer", "Homens | Alzheimer")
    km(dataset_list[4], dataset_list[5], "women-men-cancer", "Mulheres | Cancer", "Homens | Cancer")
    km(dataset_list[6], dataset_list[7], "women-men-diabetes", "Mulheres | Diabetes", "Homens | Diabetes")
    km(dataset_list[8], dataset_list[9], "women-men-heart", "Mulheres | Coracao", "Homens | Coracao")
    km(dataset_list[10], dataset_list[11], "women-men-transp", "Mulheres | Transplantes", "Homens | Transplantes")


def survival_plot_race(dataset_list):
    km(dataset_list[0], dataset_list[1], "black-white-oasis", "Negros | Oasis", "Brancos | Oasis")
    km(dataset_list[2], dataset_list[3], "black-white-alz", "Negros | Alzheimer", "Brancos | Alzheimer")
    km(dataset_list[4], dataset_list[5], "black-white-cancer", "Negros | Cancer", "Brancos | Cancer")
    km(dataset_list[6], dataset_list[7], "black-white-diabetes", "Negros | Diabetes", "Brancos | Diabetes")
    km(dataset_list[8], dataset_list[9], "black-white-heart", "Negros | Coracao", "Brancos | Coracao")
    km(dataset_list[10], dataset_list[11], "black-white-transp", "Negros | Transplantes", "Brancos | Transplantes")


def survival_plot_gender_race(dataset_list, label, gender_race_label_a, gender_race_label_b):
    km(dataset_list[0], dataset_list[1], "black-white-oasis-"+label, gender_race_label_a+" | Oasis", gender_race_label_b+" | Oasis")
    km(dataset_list[2], dataset_list[3], "black-white-alz-"+label, gender_race_label_a+" | Alzheimer", gender_race_label_b+" | Alzheimer")
    km(dataset_list[4], dataset_list[5], "black-white-cancer-"+label, gender_race_label_a+" | Cancer", gender_race_label_b+" | Cancer")
    km(dataset_list[6], dataset_list[7], "black-white-diabetes-"+label, gender_race_label_a+" | Diabetes", gender_race_label_b+" | Diabetes")
    km(dataset_list[8], dataset_list[9], "black-white-heart-"+label, gender_race_label_a+" | Coracao", gender_race_label_b+" | Coracao")
    km(dataset_list[10], dataset_list[11], "black-white-transp-"+label, gender_race_label_a+" | Transplantes", gender_race_label_b+" | Transplantes")


def main(seed, index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Conditional statistical parity
    cohort = sa_cohort.cox_neural_network()

    # First group
    # train, val, test_datasets_list_gender, labtrans = cohort_samples_fairness_gender(seed=seed, size=settings.size, cohort=cohort)

    dataset_list = cohort_samples_gender(cohort)
    logrank_stats(dataset_list)
    survival_plot_gender(dataset_list)

    # Second group
    # train, val, test_datasets_list_race, labtrans = cohort_samples_fairness_race(seed=seed, size=settings.size, cohort=cohort)

    dataset_list = cohort_samples_race(cohort)
    logrank_stats(dataset_list)
    survival_plot_race(dataset_list)

    # Third group
    # train, val, test_datasets_list_gender_race, labtrans = cohort_samples_fairness_gender_race(seed=seed, size=settings.size, cohort=cohort)

    dataset_list_w, dataset_list_m = cohort_samples_gender_race(cohort)

    logrank_stats(dataset_list_w)
    survival_plot_gender_race(dataset_list_w, "women", "Mulheres negras", "Mulheres brancas")

    logrank_stats(dataset_list_m)
    survival_plot_gender_race(dataset_list_m, "men", "Homens negros", "Homens brancos")

    # # Neural network
    # best = best_parameters.cox_time[index]
    # model = cox_time_fit_and_predict(CoxTime, train, val,
    #                                  lr=best['lr'], batch=best['batch'], dropout=best['dropout'],
    #                                  epoch=best['epoch'], weight_decay=best['weight_decay'],
    #                                  num_nodes=best['num_nodes'], shrink=best['shrink'],
    #                                  device=device, labtrans=labtrans)
    #
    # # Plot survival function
    # cox_time_survival_function(model, test_datasets_list_gender, test_datasets_list_race, test_datasets_list_gender_race)


if __name__ == "__main__":
    main(224796801, 27)
