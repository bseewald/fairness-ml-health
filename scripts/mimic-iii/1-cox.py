# Survival Analysis
#
# "Survival Analysis is used to estimate the lifespan of a particular population under study.
# It is also called ‘Time to Event’ Analysis as the goal is to estimate the time for an individual or
# a group of individuals to experience an event of interest. This time estimate is the duration
# between birth and death events. Survival Analysis was originally developed and used by Medical
# Researchers and Data Analysts to measure the lifetimes of a certain population."

import pandas as pd
import numpy as np
import time
from cohort import get_cohort as gh

import lifelines
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split


def main():

    # Get data
    cohort = gh.get_cohort()

    # Binning
    cohort['age_st'] = pd.cut(cohort['age'], np.arange(15, 91, 15))

    # Select features
    # drop = ['first_hosp_stay', 'first_icu_stay']
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age', 'level_0']
    cohort.drop(drop, axis=1, inplace=True)

    # Gender: from categorical to numerical
    cohort.gender.replace(to_replace=dict(F=1, M=0), inplace=True)
    cohort = cohort.astype({'admission_type': 'category', 'ethnicity_grouped': 'category', 'insurance': 'category',
                            'icd_alzheimer': 'category', 'icd_cancer': 'category', 'icd_diabetes': 'category', 'icd_heart': 'category',
                            'icd_transplant': 'category', 'gender': 'category', 'hospital_expire_flag': 'bool',
                            'oasis_score':'category'}, copy=False)

    cat = ['gender', 'insurance', 'ethnicity_grouped', 'admission_type', 'oasis_score', 'icd_alzheimer', 'icd_cancer',
           'icd_diabetes', 'icd_heart', 'icd_transplant', 'age_st']

    # Convert categorical variables
    cohort_df = pd.get_dummies(cohort, columns=cat, drop_first=True)

    # Datasets
    cohort_X = cohort_df[cohort_df.columns.difference(["los_hospital"])]
    cohort_y = cohort_df["los_hospital"]

    #############################################################
    # Lifelines library
    # https://github.com/CamDavidsonPilon/lifelines
    #
    # Event: hospital_expire_flag (died in hospital or not)
    # Duration: los_hospital (hospital lenght of stay -- in days)
    #############################################################

    # to-do: do 10x or 20x different random_state and use box-plot ?
    # to-do: we censored all individuals that were still under observation at time 30 ?
    random_state = 20

    # Open file
    _file = open("files/cox.txt", "a")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    # Train / test samples
    X_train, X_test, y_train, y_test = train_test_split(cohort_X, cohort_y, test_size=0.20, random_state=random_state)

    cox = sklearn_adapter(CoxPHFitter, event_col='hospital_expire_flag')
    cx = cox()

    # KFold
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

    _alphas = [100, 10, 1, 0.1, 0.01, 1e-03, 1e-04, 1e-05]
    _l1_ratios = [0, 0.001, 0.01, 0.1, 0.5]

    # Training ML model
    gcv = GridSearchCV(cx, {"penalizer": _alphas, "l1_ratio": _l1_ratios}, cv=cv)

    # Fit
    print(time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime()))
    gcv_fit = gcv.fit(X_train, y_train)

    # Score
    print(time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime()))
    gcv_score = gcv.score(X_test, y_test)


    # Best Parameters
    _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

    # C-Index
    _file.write("C-Index test sample: " + str(gcv_score) + "\n")

    cph = CoxPHFitter(penalizer=gcv_fit.best_params_['penalizer'], l1_ratio=gcv_fit.best_params_['l1_ratio'])
    cph.fit(cohort_df, duration_col="los_hospital", event_col="hospital_expire_flag")

    # Coef
    _file.write("Coeficients:\n" + str(cph.params_) + "\n\n")

    # C-Index score
    cindex = concordance_index(cohort_df['los_hospital'],
                               -cph.predict_partial_hazard(cohort_df),
                               cohort_df['hospital_expire_flag'])
    _file.write("C-Index all dataset: " + str(cindex) + "\n")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()


#################################
# FAIRNESS AND SURVIVAL ANALYSIS
#################################

# group fairness OK
# P(S > sHR | G = m) = P(S > sHR | G = f)

# group fairness NOK
# P(S > s | G = asian) = P(S > s | G = not asian)

# Conditional Statistical Parity
# P(S > s | L1 = l1, L2 = l2, E = black) = P(S > s | L1 = l1, L2 = l2, E = not black)
