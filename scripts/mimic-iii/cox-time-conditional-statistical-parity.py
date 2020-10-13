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


def cohort_samples_fairness_gender(seed, size, cohort):
    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples + legitimate factor (oasis_score, disease)

    # samples size:
    train_dataset_women_oasis = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["oasis_score"] == 3)]
    test_dataset_women_oasis = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["oasis_score"] == 3)]

    train_dataset_women_alz = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_women_alz = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_women_cancer = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["icd_cancer"] == 1)]
    test_dataset_women_cancer = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_cancer"] == 1)]

    train_dataset_women_diab = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_women_diab = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_women_heart = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["icd_heart"] == 1)]
    test_dataset_women_heart = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_heart"] == 1)]

    train_dataset_women_transp = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["icd_transplant"] == 1)]
    test_dataset_women_transp = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["icd_transplant"] == 1)]

    # samples size:
    train_dataset_men_oasis = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["oasis_score"] == 3)]
    test_dataset_men_oasis = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["oasis_score"] == 3)]

    train_dataset_men_alz = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_men_alz = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_men_cancer = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["icd_cancer"] == 1)]
    test_dataset_men_cancer = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_cancer"] == 1)]

    train_dataset_men_diab = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_men_diab = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_men_heart = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["icd_heart"] == 1)]
    test_dataset_men_heart = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_heart"] == 1)]

    train_dataset_men_transp = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["icd_transplant"] == 1)]
    test_dataset_men_transp = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["icd_transplant"] == 1)]

    # Feature transforms
    labtrans = CoxTime.label_transform()

    # preprocess input
    x_train_women_oasis, x_test_women_oasis = preprocess_input_features(train_dataset_women_oasis, test_dataset_women_oasis)
    x_train_men_oasis, x_test_men_oasis = preprocess_input_features(train_dataset_men_oasis, test_dataset_men_oasis)

    x_train_women_alz, x_test_women_alz = preprocess_input_features(train_dataset_women_alz, test_dataset_women_alz)
    x_train_men_alz, x_test_men_alz = preprocess_input_features(train_dataset_men_alz, test_dataset_men_alz)

    x_train_women_cancer, x_test_women_cancer = preprocess_input_features(train_dataset_women_cancer, test_dataset_women_cancer)
    x_train_men_cancer, x_test_men_cancer = preprocess_input_features(train_dataset_men_cancer, test_dataset_men_cancer)

    x_train_women_diab, x_test_women_diab = preprocess_input_features(train_dataset_women_diab, test_dataset_women_diab)
    x_train_men_diab, x_test_men_diab = preprocess_input_features(train_dataset_men_diab, test_dataset_men_diab)

    x_train_women_heart, x_test_women_heart = preprocess_input_features(train_dataset_women_heart, test_dataset_women_heart)
    x_train_men_heart, x_test_men_heart = preprocess_input_features(train_dataset_men_heart, test_dataset_men_heart)

    x_train_women_transp, x_test_women_transp = preprocess_input_features(train_dataset_women_transp, test_dataset_women_transp)
    x_train_men_transp, x_test_men_transp = preprocess_input_features(train_dataset_men_transp, test_dataset_men_transp)


    # preprocess target
    train_women_oasis, test_women_oasis = cox_time_preprocess_target_features(x_train_women_oasis, train_dataset_women_oasis, x_test_women_oasis, test_dataset_women_oasis, labtrans)
    train_men_oasis, test_men_oasis = cox_time_preprocess_target_features(x_train_men_oasis, train_dataset_men_oasis, x_test_men_oasis, test_dataset_men_oasis, labtrans)

    train_women_alz, test_women_alz = cox_time_preprocess_target_features(x_train_women_alz, train_dataset_women_alz, x_test_women_alz, test_dataset_women_alz, labtrans)
    train_men_alz, test_men_alz = cox_time_preprocess_target_features(x_train_men_alz, train_dataset_men_alz, x_test_men_alz, test_dataset_men_alz, labtrans)

    train_women_cancer, test_women_cancer = cox_time_preprocess_target_features(x_train_women_cancer, train_dataset_women_cancer, x_test_women_cancer, test_dataset_women_cancer, labtrans)
    train_men_cancer, test_men_cancer = cox_time_preprocess_target_features(x_train_men_cancer, train_dataset_men_cancer, x_test_men_cancer, test_dataset_men_cancer, labtrans)

    train_women_diab, test_women_diab = cox_time_preprocess_target_features(x_train_women_diab, train_dataset_women_diab, x_test_women_diab, test_dataset_women_diab, labtrans)
    train_men_diab, test_men_diab = cox_time_preprocess_target_features(x_train_men_diab, train_dataset_men_diab, x_test_men_diab, test_dataset_men_diab, labtrans)

    train_women_heart, test_women_heart = cox_time_preprocess_target_features(x_train_women_heart, train_dataset_women_heart, x_test_women_heart, test_dataset_women_heart, labtrans)
    train_men_heart, test_men_heart = cox_time_preprocess_target_features(x_train_men_heart, train_dataset_men_heart, x_test_men_heart, test_dataset_men_heart, labtrans)

    train_women_transp, test_women_transp = cox_time_preprocess_target_features(x_train_women_transp, train_dataset_women_transp, x_test_women_transp, test_dataset_women_transp, labtrans)
    train_men_transp, test_men_transp = cox_time_preprocess_target_features(x_train_men_transp, train_dataset_men_transp, x_test_men_transp, test_dataset_men_transp, labtrans)

    return test_women_oasis, test_men_oasis, test_women_alz, test_men_alz, test_women_cancer, test_men_cancer, \
           test_women_diab, test_men_diab, test_women_heart, test_men_heart, test_women_transp, test_men_transp, labtrans


def cohort_samples_fairness_race(seed, size, cohort):
    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples + legitimate factor (oasis_score, disease)
    # others: 'admission_type', 'insurance', 'age_st'

    # samples size:
    train_dataset_black_oasis = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "black") & (train_dataset["oasis_score"] == 3)]
    test_dataset_black_oasis = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["oasis_score"] == 3)]

    train_dataset_black_alz = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_black_alz = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_black_cancer = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_cancer"] == 1)]
    test_dataset_black_cancer = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_cancer"] == 1)]

    train_dataset_black_diab = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_black_diab = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_black_heart = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_heart"] == 1)]
    test_dataset_black_heart = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_heart"] == 1)]

    train_dataset_black_transp = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_transplant"] == 1)]
    test_dataset_black_transp = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_transplant"] == 1)]

    # samples size:
    train_dataset_white_oasis = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "white") & (train_dataset["oasis_score"] == 3)]
    test_dataset_white_oasis = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["oasis_score"] == 3)]

    train_dataset_white_alz = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_white_alz = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_white_cancer = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_cancer"] == 1)]
    test_dataset_white_cancer = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_cancer"] == 1)]

    train_dataset_white_diab = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_white_diab = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_white_heart = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_heart"] == 1)]
    test_dataset_white_heart = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_heart"] == 1)]

    train_dataset_white_transp = train_dataset.loc[(train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_transplant"] == 1)]
    test_dataset_white_transp = test_dataset.loc[(test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_transplant"] == 1)]

    # Feature transforms
    labtrans = CoxTime.label_transform()

    # preprocess input
    x_train_black_oasis, x_test_black_oasis = preprocess_input_features(train_dataset_black_oasis, test_dataset_black_oasis)
    x_train_white_oasis, x_test_white_oasis = preprocess_input_features(train_dataset_white_oasis, test_dataset_white_oasis)

    x_train_black_alz, x_test_black_alz = preprocess_input_features(train_dataset_black_alz, test_dataset_black_alz)
    x_train_white_alz, x_test_white_alz = preprocess_input_features(train_dataset_white_alz, test_dataset_white_alz)

    x_train_black_cancer, x_test_black_cancer = preprocess_input_features(train_dataset_black_cancer, test_dataset_black_cancer)
    x_train_white_cancer, x_test_white_cancer = preprocess_input_features(train_dataset_white_cancer, test_dataset_white_cancer)

    x_train_black_diab, x_test_black_diab = preprocess_input_features(train_dataset_black_diab, test_dataset_black_diab)
    x_train_white_diab, x_test_white_diab = preprocess_input_features(train_dataset_white_diab, test_dataset_white_diab)

    x_train_black_heart, x_test_black_heart = preprocess_input_features(train_dataset_black_heart, test_dataset_black_heart)
    x_train_white_heart, x_test_white_heart = preprocess_input_features(train_dataset_white_heart, test_dataset_white_heart)

    x_train_black_transp, x_test_black_transp = preprocess_input_features(train_dataset_black_transp, test_dataset_black_transp)
    x_train_white_transp, x_test_white_transp = preprocess_input_features(train_dataset_white_transp, test_dataset_white_transp)


    # preprocess target
    train_black_oasis, test_black_oasis = cox_time_preprocess_target_features(x_train_black_oasis, train_dataset_black_oasis, x_test_black_oasis, test_dataset_black_oasis, labtrans)
    train_white_oasis, test_white_oasis = cox_time_preprocess_target_features(x_train_white_oasis, train_dataset_white_oasis, x_test_white_oasis, test_dataset_white_oasis, labtrans)

    train_black_alz, test_black_alz = cox_time_preprocess_target_features(x_train_black_alz, train_dataset_black_alz, x_test_black_alz, test_dataset_black_alz, labtrans)
    train_white_alz, test_white_alz = cox_time_preprocess_target_features(x_train_white_alz, train_dataset_white_alz, x_test_white_alz, test_dataset_white_alz, labtrans)

    train_black_cancer, test_black_cancer = cox_time_preprocess_target_features(x_train_black_cancer, train_dataset_black_cancer, x_test_black_cancer, test_dataset_black_cancer, labtrans)
    train_white_cancer, test_white_cancer = cox_time_preprocess_target_features(x_train_white_cancer, train_dataset_white_cancer, x_test_white_cancer, test_dataset_white_cancer, labtrans)

    train_black_diab, test_black_diab = cox_time_preprocess_target_features(x_train_black_diab, train_dataset_black_diab, x_test_black_diab, test_dataset_black_diab, labtrans)
    train_white_diab, test_white_diab = cox_time_preprocess_target_features(x_train_white_diab, train_dataset_white_diab, x_test_white_diab, test_dataset_white_diab, labtrans)

    train_black_heart, test_black_heart = cox_time_preprocess_target_features(x_train_black_heart, train_dataset_black_heart, x_test_black_heart, test_dataset_black_heart, labtrans)
    train_white_heart, test_white_heart = cox_time_preprocess_target_features(x_train_white_heart, train_dataset_white_heart, x_test_white_heart, test_dataset_white_heart, labtrans)

    train_black_transp, test_black_transp = cox_time_preprocess_target_features(x_train_black_transp, train_dataset_black_transp, x_test_black_transp, test_dataset_black_transp, labtrans)
    train_white_transp, test_white_transp = cox_time_preprocess_target_features(x_train_white_transp, train_dataset_white_transp, x_test_white_transp, test_dataset_white_transp, labtrans)

    return test_black_oasis, test_white_oasis, test_black_alz, test_white_alz, test_black_cancer, test_white_cancer, \
           test_black_diab, test_white_diab, test_black_heart, test_white_heart, test_black_transp, test_white_transp, labtrans


def cohort_samples_fairness_gender_race(seed, size, cohort):
   # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Fairness samples + legitimate factor (oasis_score, disease)
    # others: 'admission_type', 'insurance', 'age_st'

    # samples size:
    train_dataset_women_black_oasis = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["oasis_score"] == 3)]
    test_dataset_women_black_oasis = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["oasis_score"] == 3)]

    train_dataset_women_black_alz = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_women_black_alz = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_women_black_cancer = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_cancer"] == 1)]
    test_dataset_women_black_cancer = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_cancer"] == 1)]

    train_dataset_women_black_diab = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_women_black_diab = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_women_black_heart = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_heart"] == 1)]
    test_dataset_women_black_heart = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_heart"] == 1)]

    train_dataset_women_black_transp = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_transplant"] == 1)]
    test_dataset_women_black_transp = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_transplant"] == 1)]

    # samples size:
    train_dataset_women_white_oasis = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["oasis_score"] == 3)]
    test_dataset_women_white_oasis = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["oasis_score"] == 3)]

    train_dataset_women_white_alz = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_women_white_alz = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_women_white_cancer = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_cancer"] == 1)]
    test_dataset_women_white_cancer = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_cancer"] == 1)]

    train_dataset_women_white_diab = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_women_white_diab = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_women_white_heart = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_heart"] == 1)]
    test_dataset_women_white_heart = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_heart"] == 1)]

    train_dataset_women_white_transp = train_dataset.loc[(train_dataset["gender"] == 1) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_transplant"] == 1)]
    test_dataset_women_white_transp = test_dataset.loc[(test_dataset["gender"] == 1) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_transplant"] == 1)]

    # samples size:
    train_dataset_men_black_oasis = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["oasis_score"] == 3)]
    test_dataset_men_black_oasis = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["oasis_score"] == 3)]

    train_dataset_men_black_alz = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_men_black_alz = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_men_black_cancer = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_cancer"] == 1)]
    test_dataset_men_black_cancer = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_cancer"] == 1)]

    train_dataset_men_black_diab = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_men_black_diab = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_men_black_heart = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_heart"] == 1)]
    test_dataset_men_black_heart = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_heart"] == 1)]

    train_dataset_men_black_transp = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "black") & (train_dataset["icd_transplant"] == 1)]
    test_dataset_men_black_transp = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "black") & (test_dataset["icd_transplant"] == 1)]

    # samples size:
    train_dataset_men_white_oasis = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["oasis_score"] == 3)]
    test_dataset_men_white_oasis = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["oasis_score"] == 3)]

    train_dataset_men_white_alz = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_alzheimer"] == 1)]
    test_dataset_men_white_alz = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_alzheimer"] == 1)]

    train_dataset_men_white_cancer = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_cancer"] == 1)]
    test_dataset_men_white_cancer = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_cancer"] == 1)]

    train_dataset_men_white_diab = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_diabetes"] == 1)]
    test_dataset_men_white_diab = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_diabetes"] == 1)]

    train_dataset_men_white_heart = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_heart"] == 1)]
    test_dataset_men_white_heart = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_heart"] == 1)]

    train_dataset_men_white_transp = train_dataset.loc[(train_dataset["gender"] == 0) & (train_dataset["ethnicity_grouped"] == "white") & (train_dataset["icd_transplant"] == 1)]
    test_dataset_men_white_transp = test_dataset.loc[(test_dataset["gender"] == 0) & (test_dataset["ethnicity_grouped"] == "white") & (test_dataset["icd_transplant"] == 1)]

    # Feature transforms
    labtrans = CoxTime.label_transform()

    # preprocess input
    x_train_women_black_oasis, x_test_women_black_oasis = preprocess_input_features(train_dataset_women_black_oasis, test_dataset_women_black_oasis)
    x_train_women_white_oasis, x_test_women_white_oasis = preprocess_input_features(train_dataset_women_white_oasis, test_dataset_women_white_oasis)

    x_train_women_black_alz, x_test_women_black_alz = preprocess_input_features(train_dataset_women_black_alz, test_dataset_women_black_alz)
    x_train_women_white_alz, x_test_women_white_alz = preprocess_input_features(train_dataset_women_white_alz, test_dataset_women_white_alz)

    x_train_women_black_cancer, x_test_women_black_cancer = preprocess_input_features(train_dataset_women_black_cancer, test_dataset_women_black_cancer)
    x_train_women_white_cancer, x_test_women_white_cancer = preprocess_input_features(train_dataset_women_white_cancer, test_dataset_women_white_cancer)

    x_train_women_black_diab, x_test_women_black_diab = preprocess_input_features(train_dataset_women_black_diab, test_dataset_women_black_diab)
    x_train_women_white_diab, x_test_women_white_diab = preprocess_input_features(train_dataset_women_white_diab, test_dataset_women_white_diab)

    x_train_women_black_heart, x_test_women_black_heart = preprocess_input_features(train_dataset_women_black_heart, test_dataset_women_black_heart)
    x_train_women_white_heart, x_test_women_white_heart = preprocess_input_features(train_dataset_women_white_heart, test_dataset_women_white_heart)

    x_train_women_black_transp, x_test_women_black_transp = preprocess_input_features(train_dataset_women_black_transp, test_dataset_women_black_transp)
    x_train_women_white_transp, x_test_women_white_transp = preprocess_input_features(train_dataset_women_white_transp, test_dataset_women_white_transp)

    x_train_men_black_oasis, x_test_men_black_oasis = preprocess_input_features(train_dataset_men_black_oasis, test_dataset_men_black_oasis)
    x_train_men_white_oasis, x_test_men_white_oasis = preprocess_input_features(train_dataset_men_white_oasis, test_dataset_men_white_oasis)

    x_train_men_black_alz, x_test_men_black_alz = preprocess_input_features(train_dataset_men_black_alz, test_dataset_men_black_alz)
    x_train_men_white_alz, x_test_men_white_alz = preprocess_input_features(train_dataset_men_white_alz, test_dataset_men_white_alz)

    x_train_men_black_cancer, x_test_men_black_cancer = preprocess_input_features(train_dataset_men_black_cancer, test_dataset_men_black_cancer)
    x_train_men_white_cancer, x_test_men_white_cancer = preprocess_input_features(train_dataset_men_white_cancer, test_dataset_men_white_cancer)

    x_train_men_black_diab, x_test_men_black_diab = preprocess_input_features(train_dataset_men_black_diab, test_dataset_men_black_diab)
    x_train_men_white_diab, x_test_men_white_diab = preprocess_input_features(train_dataset_men_white_diab, test_dataset_men_white_diab)

    x_train_men_black_heart, x_test_men_black_heart = preprocess_input_features(train_dataset_men_black_heart, test_dataset_men_black_heart)
    x_train_men_white_heart, x_test_men_white_heart = preprocess_input_features(train_dataset_men_white_heart, test_dataset_men_white_heart)

    x_train_men_black_transp, x_test_men_black_transp = preprocess_input_features(train_dataset_men_black_transp, test_dataset_men_black_transp)
    x_train_men_white_transp, x_test_men_white_transp = preprocess_input_features(train_dataset_men_white_transp, test_dataset_men_white_transp)


    # preprocess target
    train_women_black_oasis, test_women_black_oasis = cox_time_preprocess_target_features(x_train_women_black_oasis, train_dataset_women_black_oasis, x_test_women_black_oasis, test_dataset_women_black_oasis, labtrans)
    train_women_white_oasis, test_women_white_oasis = cox_time_preprocess_target_features(x_train_women_white_oasis, train_dataset_women_white_oasis, x_test_women_white_oasis, test_dataset_women_white_oasis, labtrans)

    train_women_black_alz, test_women_black_alz = cox_time_preprocess_target_features(x_train_women_black_alz, train_dataset_women_black_alz, x_test_women_black_alz, test_dataset_women_black_alz, labtrans)
    train_women_white_alz, test_women_white_alz = cox_time_preprocess_target_features(x_train_women_white_alz, train_dataset_women_white_alz, x_test_women_white_alz, test_dataset_women_white_alz, labtrans)

    train_women_black_cancer, test_women_black_cancer = cox_time_preprocess_target_features(x_train_women_black_cancer, train_dataset_women_black_cancer, x_test_women_black_cancer, test_dataset_women_black_cancer, labtrans)
    train_women_white_cancer, test_women_white_cancer = cox_time_preprocess_target_features(x_train_women_white_cancer, train_dataset_women_white_cancer, x_test_women_white_cancer, test_dataset_women_white_cancer, labtrans)

    train_women_black_diab, test_women_black_diab = cox_time_preprocess_target_features(x_train_women_black_diab, train_dataset_women_black_diab, x_test_women_black_diab, test_dataset_women_black_diab, labtrans)
    train_women_white_diab, test_women_white_diab = cox_time_preprocess_target_features(x_train_women_white_diab, train_dataset_women_white_diab, x_test_women_white_diab, test_dataset_women_white_diab, labtrans)

    train_women_black_heart, test_women_black_heart = cox_time_preprocess_target_features(x_train_women_black_heart, train_dataset_women_black_heart, x_test_women_black_heart, test_dataset_women_black_heart, labtrans)
    train_women_white_heart, test_women_white_heart = cox_time_preprocess_target_features(x_train_women_white_heart, train_dataset_women_white_heart, x_test_women_white_heart, test_dataset_women_white_heart, labtrans)

    train_women_black_transp, test_women_black_transp = cox_time_preprocess_target_features(x_train_women_black_transp, train_dataset_women_black_transp, x_test_women_black_transp, test_dataset_women_black_transp, labtrans)
    train_women_white_transp, test_women_white_transp = cox_time_preprocess_target_features(x_train_women_white_transp, train_dataset_women_white_transp, x_test_women_white_transp, test_dataset_women_white_transp, labtrans)

    train_men_black_oasis, test_men_black_oasis = cox_time_preprocess_target_features(x_train_men_black_oasis, train_dataset_men_black_oasis, x_test_men_black_oasis, test_dataset_men_black_oasis, labtrans)
    train_men_white_oasis, test_men_white_oasis = cox_time_preprocess_target_features(x_train_men_white_oasis, train_dataset_men_white_oasis, x_test_men_white_oasis, test_dataset_men_white_oasis, labtrans)

    train_men_black_alz, test_men_black_alz = cox_time_preprocess_target_features(x_train_men_black_alz, train_dataset_men_black_alz, x_test_men_black_alz, test_dataset_men_black_alz, labtrans)
    train_men_white_alz, test_men_white_alz = cox_time_preprocess_target_features(x_train_men_white_alz, train_dataset_men_white_alz, x_test_men_white_alz, test_dataset_men_white_alz, labtrans)

    train_men_black_cancer, test_men_black_cancer = cox_time_preprocess_target_features(x_train_men_black_cancer, train_dataset_men_black_cancer, x_test_men_black_cancer, test_dataset_men_black_cancer, labtrans)
    train_men_white_cancer, test_men_white_cancer = cox_time_preprocess_target_features(x_train_men_white_cancer, train_dataset_men_white_cancer, x_test_men_white_cancer, test_dataset_men_white_cancer, labtrans)

    train_men_black_diab, test_men_black_diab = cox_time_preprocess_target_features(x_train_men_black_diab, train_dataset_men_black_diab, x_test_men_black_diab, test_dataset_men_black_diab, labtrans)
    train_men_white_diab, test_men_white_diab = cox_time_preprocess_target_features(x_train_men_white_diab, train_dataset_men_white_diab, x_test_men_white_diab, test_dataset_men_white_diab, labtrans)

    train_men_black_heart, test_men_black_heart = cox_time_preprocess_target_features(x_train_men_black_heart, train_dataset_men_black_heart, x_test_men_black_heart, test_dataset_men_black_heart, labtrans)
    train_men_white_heart, test_men_white_heart = cox_time_preprocess_target_features(x_train_men_white_heart, train_dataset_men_white_heart, x_test_men_white_heart, test_dataset_men_white_heart, labtrans)

    train_men_black_transp, test_men_black_transp = cox_time_preprocess_target_features(x_train_men_black_transp, train_dataset_men_black_transp, x_test_men_black_transp, test_dataset_men_black_transp, labtrans)
    train_men_white_transp, test_men_white_transp = cox_time_preprocess_target_features(x_train_men_white_transp, train_dataset_men_white_transp, x_test_men_white_transp, test_dataset_men_white_transp, labtrans)

    return test_women_black_oasis, test_women_white_oasis, test_women_black_alz, test_women_white_alz, test_women_black_cancer, test_women_white_cancer, \
           test_women_black_diab, test_women_white_diab, test_women_black_heart, test_women_white_heart, test_women_black_transp, test_women_white_transp, \
           test_men_black_oasis, test_men_white_oasis, test_men_black_alz, test_men_white_alz, test_men_black_cancer, test_men_white_cancer, \
           test_men_black_diab, test_men_white_diab, test_men_black_heart, test_men_white_heart, test_men_black_transp, test_men_white_transp, labtrans


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
    fig_path = "img/cox-time/conditional-statistical-parity/cox-time-conditional-statistical-parity-"
    ax.get_figure().savefig(fig_path + group_name + "-" + fig_time + ".png", format="png", bbox_inches="tight", dpi=600)
    plt.close()


def main(seed, file, index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Conditional statistical parity
    cohort = sa_cohort.cox_neural_network()

    # First group
    test_women_oasis, test_men_oasis, test_women_alz, test_men_alz, test_women_cancer, test_men_cancer, \
    test_women_diab, test_men_diab, test_women_heart, test_men_heart, test_women_transp, test_men_transp, \
    labtrans = cohort_samples_fairness_gender(seed=seed, size=settings.size, cohort=cohort)

    # Second group
    test_black_oasis, test_white_oasis, test_black_alz, test_white_alz, test_black_cancer, test_white_cancer, \
    test_black_diab, test_white_diab, test_black_heart, test_white_heart, test_black_transp, test_white_transp, \
    labtrans2 = cohort_samples_fairness_race(seed=seed, size=settings.size, cohort=cohort)

    # Third group
    test_women_black_oasis, test_women_white_oasis, test_women_black_alz, test_women_white_alz, test_women_black_cancer, test_women_white_cancer, \
    test_women_black_diab, test_women_white_diab, test_women_black_heart, test_women_white_heart, test_women_black_transp, test_women_white_transp, \
    test_men_black_oasis, test_men_white_oasis, test_men_black_alz, test_men_white_alz, test_men_black_cancer, test_men_white_cancer, \
    test_men_black_diab, test_men_white_diab, test_men_black_heart, test_men_white_heart, test_men_black_transp, test_men_white_transp, \
    labtrans3 = cohort_samples_fairness_gender_race(seed=seed, size=settings.size, cohort=cohort)

    # Reload model
    best = best_parameters.cox_time[index]
    model = cox_time_reload_model(file, weight_decay=best['weight_decay'], shrink=best['shrink'], device=device)

    # Predict survival curve from first group
    surv_women_oasis = model.predict_surv_df(test_women_oasis[0])
    surv_men_oasis = model.predict_surv_df(test_men_oasis[0])
    surv_women_alz = model.predict_surv_df(test_women_alz[0])
    surv_men_alz = model.predict_surv_df(test_men_alz[0])
    surv_women_cancer = model.predict_surv_df(test_women_cancer[0])
    surv_men_cancer = model.predict_surv_df(test_men_cancer[0])
    surv_women_diab = model.predict_surv_df(test_women_diab[0])
    surv_men_diab = model.predict_surv_df(test_men_diab[0])
    surv_women_heart = model.predict_surv_df(test_women_heart[0])
    surv_men_heart = model.predict_surv_df(test_men_heart[0])
    surv_women_transp = model.predict_surv_df(test_women_transp[0])
    surv_men_transp = model.predict_surv_df(test_men_transp[0])

    # Plotting survival curve from first group
    survival_curve_plot(surv_women_oasis, surv_men_oasis, "women | oasis score", "men | oasis score", "women-men-oasis")
    survival_curve_plot(surv_women_alz, surv_men_alz, "women | alzheimer", "men | alzheimer", "women-men-alzheimer")
    survival_curve_plot(surv_women_cancer, surv_men_cancer, "women | cancer", "men | cancer", "women-men-cancer")
    survival_curve_plot(surv_women_diab, surv_men_diab, "women | diabetes", "men | diabetes", "women-men-diabetes")
    survival_curve_plot(surv_women_heart, surv_men_heart, "women | heart", "men | heart", "women-men-heart")
    survival_curve_plot(surv_women_transp, surv_men_transp, "women | transplant", "men | transplant", "women-men-transplant")

    # Predict survival curve from second group
    surv_black_oasis = model.predict_surv_df(test_black_oasis[0])
    surv_white_oasis = model.predict_surv_df(test_white_oasis[0])
    surv_black_alz = model.predict_surv_df(test_black_alz[0])
    surv_white_alz = model.predict_surv_df(test_white_alz[0])
    surv_black_cancer = model.predict_surv_df(test_black_cancer[0])
    surv_white_cancer = model.predict_surv_df(test_white_cancer[0])
    surv_black_diab = model.predict_surv_df(test_black_diab[0])
    surv_white_diab = model.predict_surv_df(test_white_diab[0])
    surv_black_heart = model.predict_surv_df(test_black_heart[0])
    surv_white_heart = model.predict_surv_df(test_white_heart[0])
    surv_black_transp = model.predict_surv_df(test_black_transp[0])
    surv_white_transp = model.predict_surv_df(test_white_transp[0])

    # Predict survival curve from second group
    survival_curve_plot(surv_black_oasis, surv_white_oasis, "black | oasis score", "white | oasis score", "black-white-oasis")
    survival_curve_plot(surv_black_alz, surv_white_alz, "black | alzheimer", "white | alzheimer", "black-white-alzheimer")
    survival_curve_plot(surv_black_cancer, surv_white_cancer, "black | cancer", "white | cancer", "black-white-cancer")
    survival_curve_plot(surv_black_diab, surv_white_diab, "black | diabetes", "white | diabetes", "black-white-diabetes")
    survival_curve_plot(surv_black_heart, surv_white_heart, "black | heart", "white | heart", "black-white-heart")
    survival_curve_plot(surv_black_transp, surv_white_transp, "black | transplant", "white | transplant", "black-white-transplant")

    # Predict survival curve from third group
    surv_women_black_oasis = model.predict_surv_df(test_women_black_oasis[0])
    surv_women_white_oasis = model.predict_surv_df(test_women_white_oasis[0])
    surv_women_black_alz = model.predict_surv_df(test_women_black_alz[0])
    surv_women_white_alz = model.predict_surv_df(test_women_white_alz[0])
    surv_women_black_cancer = model.predict_surv_df(test_women_black_cancer[0])
    surv_women_white_cancer = model.predict_surv_df(test_women_white_cancer[0])
    surv_women_black_diab = model.predict_surv_df(test_women_black_diab[0])
    surv_women_white_diab = model.predict_surv_df(test_women_white_diab[0])
    surv_women_black_heart = model.predict_surv_df(test_women_black_heart[0])
    surv_women_white_heart = model.predict_surv_df(test_women_white_heart[0])
    surv_women_black_transp = model.predict_surv_df(test_women_black_transp[0])
    surv_women_white_transp = model.predict_surv_df(test_women_white_transp[0])

    # Predict survival curve from third group
    survival_curve_plot(surv_women_black_oasis, surv_women_white_oasis, "black women | oasis score", "white women | oasis score", "black-women-white-women-oasis")
    survival_curve_plot(surv_women_black_alz, surv_women_white_alz, "black women | alzheimer", "white women | alzheimer", "black-women-white-women-alzheimer")
    survival_curve_plot(surv_women_black_cancer, surv_women_white_cancer, "black women | cancer", "white women | cancer", "black-women-white-women-cancer")
    survival_curve_plot(surv_women_black_diab, surv_women_white_diab, "black women | diabetes", "white women | diabetes", "black-women-white-women-diabetes")
    survival_curve_plot(surv_women_black_heart, surv_women_white_heart, "black women | heart", "white women | heart", "black-women-white-women-heart")
    survival_curve_plot(surv_women_black_transp, surv_women_white_transp, "black women | transplant", "white women | transplant", "black-women-white-women-transplant")

    # Predict survival curve from fourth group
    surv_men_black_oasis = model.predict_surv_df(test_men_black_oasis[0])
    surv_men_white_oasis = model.predict_surv_df(test_men_white_oasis[0])
    surv_men_black_alz = model.predict_surv_df(test_men_black_alz[0])
    surv_men_white_alz = model.predict_surv_df(test_men_white_alz[0])
    surv_men_black_cancer = model.predict_surv_df(test_men_black_cancer[0])
    surv_men_white_cancer = model.predict_surv_df(test_men_white_cancer[0])
    surv_men_black_diab = model.predict_surv_df(test_men_black_diab[0])
    surv_men_white_diab = model.predict_surv_df(test_men_white_diab[0])
    surv_men_black_heart = model.predict_surv_df(test_men_black_heart[0])
    surv_men_white_heart = model.predict_surv_df(test_men_white_heart[0])
    surv_men_black_transp = model.predict_surv_df(test_men_black_transp[0])
    surv_men_white_transp = model.predict_surv_df(test_men_white_transp[0])

    # Predict survival curve from fourth group
    survival_curve_plot(surv_men_black_oasis, surv_men_white_oasis, "black men | oasis score", "white men | oasis score", "black-men-white-men-oasis")
    survival_curve_plot(surv_men_black_alz, surv_men_white_alz, "black men | alzheimer", "white men | alzheimer", "black-men-white-men-alzheimer")
    survival_curve_plot(surv_men_black_cancer, surv_men_white_cancer, "black men | cancer", "white men | cancer", "black-men-white-men-cancer")
    survival_curve_plot(surv_men_black_diab, surv_men_white_diab, "black men | diabetes", "white men | diabetes", "black-men-white-men-diabetes")
    survival_curve_plot(surv_men_black_heart, surv_men_white_heart, "black men | heart", "white men | heart", "black-men-white-men-heart")
    survival_curve_plot(surv_men_black_transp, surv_men_white_transp, "black men | transplant", "white men | transplant", "black-men-white-men-transplant")


if __name__ == "__main__":
    # second best seed --> 224796801 (n.27)
    files = sorted(glob.glob(settings.PATH + "*.pt"), key=os.path.getmtime)
    main(224796801, files[27], 27)

    # i = 0
    # for seed in settings.seed:
    #     main(seed, files[i], i)
    #     i+=1
