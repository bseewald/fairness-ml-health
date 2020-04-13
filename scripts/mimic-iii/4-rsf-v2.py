import pandas as pd
import numpy as np
import psycopg2
import time
from cohort import get_cohort as gh

from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

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
                            'icd_alzheimer': 'bool', 'icd_cancer': 'bool', 'icd_diabetes': 'bool', 'icd_heart': 'bool',
                            'icd_transplant': 'bool', 'gender': 'bool', 'hospital_expire_flag': 'bool',
                            'oasis_score':'category'}, copy=False)

    # Select features
    cohort_X = cohort[cohort.columns.difference(["los_hospital", "hospital_expire_flag"])]

    cohort_y = cohort[["hospital_expire_flag", "los_hospital"]]
    # cohort_y = cohort_y.astype({'hospital_expire_flag': 'bool'}, copy=False)
    cohort_y = Surv.from_dataframe("hospital_expire_flag", "los_hospital", cohort_y)


    #############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # Random Survival Forest
    #
    #############################################################

    random_state = 20
    old_score = 0

    # Open file
    _file = open("files/cox-rsf.txt", "a")

    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    # Transformation
    Xt = OneHotEncoder().fit_transform(cohort_X)
    Xt = np.column_stack(Xt.values)
    feature_names = cohort_X.columns.tolist()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(Xt.transpose(), cohort_y, test_size=0.20, random_state=random_state)

    # KFold
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

    # Params
    split = [2, 4, 6, 8]
    leaf = [8, 10, 20, 50] # v2
    n_estimators = [500, 1000]
    max_features = ["auto", "sqrt"]
    n_jobs = [-1]
    random_state_rsf = [random_state]

    params = {'n_estimators': n_estimators, 'min_samples_split': split, 'min_samples_leaf': leaf,
              'max_features': max_features, 'n_jobs': n_jobs, 'random_state': random_state_rsf}

    # Train model
    rsf = RandomSurvivalForest()
    gcv = GridSearchCV(rsf, param_grid=params, cv=cv)

    print(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
    gcv_fit = gcv.fit(X_train, y_train)

    # C-index score
    print(time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime()))
    gcv_score = gcv.score(X_test, y_test)

    if gcv_score > old_score:

        old_score = gcv_score

        print(gcv_fit.best_params_)

        # Best Parameters
        _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

        # C-Index
        _file.write("C-Index: " + str(gcv_score) + "\n")

    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    _file.write("\n*** The last one is the best configuration! ***\n\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()