import pandas as pd
import numpy as np
import psycopg2
import time
from time import gmtime, strftime

from sklearn.model_selection import train_test_split

from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

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
    # Random Survival Forest
    #
    #############################################################

    # Transformation
    Xt = OneHotEncoder().fit_transform(cohort_X)
    Xt = np.column_stack((Xt.values))
    feature_names = cohort_X.columns.tolist()

    # Train / test split
    random_state = 20
    X_train, X_test, y_train, y_test = train_test_split(Xt.transpose(), cohort_y, test_size=0.25, random_state=random_state)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("init: " + time_string)

    # Train model
    rsf = RandomSurvivalForest(n_estimators=1000,
                            min_samples_split=10,
                            min_samples_leaf=15,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=random_state)
    rsf.fit(X_train, y_train)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("final: " + time_string)

    # C-index score
    rsf.score(X_test, y_test)

    # TO-DO: ML!


if __name__ == "__main__":
    main()