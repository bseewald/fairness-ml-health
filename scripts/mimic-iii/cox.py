# Survival Analysis
#
# "Survival Analysis is used to estimate the lifespan of a particular population under study.
# It is also called ‘Time to Event’ Analysis as the goal is to estimate the time for an individual or
# a group of individuals to experience an event of interest. This time estimate is the duration
# between birth and death events. Survival Analysis was originally developed and used by Medical
# Researchers and Data Analysts to measure the lifetimes of a certain population."

import pandas as pd
import numpy as np
import psycopg2
import time
from time import gmtime, strftime

import lifelines
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.utils import k_fold_cross_validation


# Classical Cox
def cox_regression(df, duration, event, penalizer, strata_df=None):
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=duration, event_col=event, strata=strata_df, show_progress=True, step_size=0.50)
    return cph


def main():

    #######################
    # POSTGRESQL Connection
    #######################
    host = '/tmp'
    user='postgres'
    passwd='postgres'
    con = psycopg2.connect(dbname ='mimic', user=user, password=passwd, host=host)
    cur = con.cursor()

    # Cohort Table
    cohort_query = 'SELECT * FROM mimiciii.cohort_survival'
    cohort = pd.read_sql_query(cohort_query, con)

    # Binning
    cohort['age_st'] = pd.cut(cohort['age'], np.arange(15, 91, 15))

    # Select features
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity',
            'hospstay_seq', 'first_hosp_stay', 'intime', 'outtime', 'los_icu', 'icustay_seq', 'first_icu_stay', 'row_id',
            'seq_num', 'icd9_code', 'age']
    cohort_class = cohort.drop(drop, axis=1)

    cat = ['gender', 'insurance', 'ethnicity_grouped', 'admission_type', 'oasis_score', 'icd_alzheimer', 'icd_cancer',
        'icd_diabetes', 'icd_heart', 'icd_transplant', 'age_st']


    #############################################################
    # Lifelines library
    # https://github.com/CamDavidsonPilon/lifelines
    #
    # Event: hospital_expire_flag (died in hospital or not)
    # Duration: los_hospital (hospital lenght of stay -- in days)
    #############################################################

    # Convert categorical variables
    cohort_df = pd.get_dummies(cohort_class, columns=cat, drop_first=True)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("init: " + time_string)

    # Training model
    cx = cox_regression(cohort_df, 'los_hospital', 'hospital_expire_flag', penalizer=0)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("final: " + time_string)

    # C-Index score
    cindex = concordance_index(cohort_df['los_hospital'],
                     -cx.predict_partial_hazard(cohort_df),
                     cohort_df['hospital_expire_flag'])
    print(cindex)

    # TO-DO: ML
    # scores = k_fold_cross_validation(model, dataset, 'T', event_col='E', k=10)
    # print(np.mean(scores))

    # def mae(Y_true, Y_pred):
    #     return np.abs(np.subtract(Y_true, Y_pred)).mean()

    # for p in [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    #     print(p)
    #     cx = CoxPHFitter(penalizer=p)
    #     scores = k_fold_cross_validation(cx, df_model, duration_col='los_hospital',
    #                                      event_col='hospital_expire_flag', evaluation_measure=mae)
    #     print(np.mean(scores))

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
