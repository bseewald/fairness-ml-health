import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from cohort import get_cohort as gh

from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split


def main():

    # Get data
    cohort = gh.get_cohort()

    # Binning
    cohort['age_st'] = pd.cut(cohort['age'], np.arange(15, 91, 15))

    # Neural network
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age', 'level_0']
    cohort.drop(drop, axis=1, inplace=True)

    # Gender: from categorical to numerical
    cohort.gender.replace(to_replace=dict(F=1, M=0), inplace=True)
    cohort = cohort.astype({'admission_type': 'category', 'ethnicity_grouped': 'category', 'insurance': 'category',
                            'icd_alzheimer': 'int64', 'icd_cancer': 'int64', 'icd_diabetes': 'int64', 'icd_heart': 'int64',
                            'icd_transplant': 'int64', 'gender': 'int64', 'hospital_expire_flag': 'int64',
                            'oasis_score':'int64'}, copy=False)

    # Select features
    cohort_y = cohort[["hospital_expire_flag", "los_hospital"]]
    cohort_y['hospital_expire_flag'] = cohort_y['hospital_expire_flag'].astype(bool)
    cohort_y = Surv.from_dataframe("hospital_expire_flag", "los_hospital", cohort_y)

    cohort_X = cohort[cohort.columns.difference(["los_hospital", "hospital_expire_flag"])]

    ############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # CoxnetSurvivalAnalysis
    #
    ############################################################

    random_state = 20
    old_score = 0

    # Open file
    _file = open("files/cox-net.txt", "a")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    # OneHot
    cohort_Xt = OneHotEncoder().fit_transform(cohort_X)

    # Train / test samples
    X_train, X_test, y_train, y_test = train_test_split(cohort_Xt, cohort_y, test_size=0.20, random_state=random_state)

    # KFold
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

    _alphas = [100, 10, 1, 0.1, 0.01, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
    _l1_ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.01, 0.001]

    _file.write("Tuning hyper-parameters\n\n")
    _file.write("Alphas: " + str(_alphas) + "\n\n")

    # Training Model
    for ratio in _l1_ratios:
        coxnet = CoxnetSurvivalAnalysis(alphas=_alphas, l1_ratio=ratio).fit(cohort_Xt, cohort_y)

        _file.write("\nL1 Ratio: " + str(ratio) + "\n")

        # GridSearchCV
        gcv = GridSearchCV(coxnet, {"alphas": [[v] for v in coxnet.alphas_]}, cv=cv)

        # Fit
        gcv_fit = gcv.fit(X_train, y_train)

        # Score
        gcv_score = gcv.score(X_test, y_test)

        if gcv_score > old_score:

            old_score = gcv_score

            # Results
            results = pd.DataFrame(gcv_fit.cv_results_)

            alphas = results.param_alphas.map(lambda x: x[0])
            mean = results.mean_test_score
            std = results.std_test_score

            # Plot
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(alphas, mean)
            ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
            ax.set_xscale("log")
            ax.set_ylabel("c-index")
            ax.set_xlabel("alpha")
            ax.axvline(gcv.best_params_['alphas'][0], c='C1')
            ax.grid(True)
            fig.savefig("img/coxnet.png", format="png", bbox_inches="tight")

            # Best Parameters
            _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

            # C-Index
            _file.write("C-Index: " + str(gcv_score) + "\n")

            # Coef
            coef = pd.Series(gcv_fit.best_estimator_.coef_[:, 0], index=cohort_Xt.columns)
            _file.write("Coeficients:\n" + str(coef[coef != 0]) + "\n\n")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    _file.write("\n*** The last one is the best configuration! ***\n\n")

    # Close file
    _file.close()

if __name__ == "__main__":
    main()