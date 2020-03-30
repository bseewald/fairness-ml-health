import pandas as pd
import numpy as np
import psycopg2
import time
import matplotlib.pyplot as plt
from cohort import get_cohort as gh

from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def main():

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

    # Select features
    cohort_y = cohort[["hospital_expire_flag", "los_hospital"]]
    cohort_y['hospital_expire_flag'] = cohort_y['hospital_expire_flag'].astype(bool)
    cohort_y = Surv.from_dataframe("hospital_expire_flag", "los_hospital", cohort_y)

    cohort_X = cohort[cohort.columns.difference(["los_hospital", "hospital_expire_flag"])]
    cohort_X = cohort_X.astype({'admission_type': 'category', 'ethnicity_grouped': 'category',
                                 'gender': 'category', 'insurance': 'category', 'icd_alzheimer': 'category',
                                 'icd_cancer': 'category', 'icd_diabetes': 'category', 'icd_heart': 'category',
                                 'icd_transplant': 'category'}, copy=False)


    #############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # CoxnetSurvivalAnalysis
    #
    #############################################################

    # open file
    _file = open("files/cox-net.txt", "a")

    named_tuple = time.localtime()
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    _file.write("init: " + time_string + "\n")

    random_state = 20
    cohort_Xt = OneHotEncoder().fit_transform(cohort_X)

    # Can I use it ? How ?
    # X_train, X_test, y_train, y_test = train_test_split(Xt, cohort_y, test_size=0.25, random_state=random_state)

    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)
    _alphas = [1e+04, 1e+03, 100, 10, 1, 0.1, 0.01, 1e-03, 1e-04, 1e-05, 1e-06, 0]
    coxnet = CoxnetSurvivalAnalysis(alphas=_alphas, l1_ratio=1.0).fit(cohort_Xt, cohort_y)
    _file.write(str(coxnet.alphas_) + "\n")
    # CoxnetSurvivalAnalysis(l1_ratio=1e-16, tol=1e-09, normalize=False)
    gcv = GridSearchCV(coxnet, {"alphas": [[v] for v in coxnet.alphas_]}, cv=cv).fit(cohort_Xt, cohort_y)

    named_tuple = time.localtime()
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    _file.write("final: " + time_string + "\n")

    # Results
    results = pd.DataFrame(gcv.cv_results_)

    alphas = results.param_alphas.map(lambda x: x[0])
    mean = results.mean_test_score
    std = results.std_test_score

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_['alphas'][0], c='C1')
    ax.grid(True)
    fig.savefig("img/coxnet.png", format="png", bbox_inches="tight")

    coef = pd.Series(gcv.best_estimator_.coef_[:, 0], index=cohort_Xt.columns)
    _file.write(str(coef[coef != 0]) + "\n")

    _file.close()

if __name__ == "__main__":
    main()