import psycopg2
import pandas as pd
import numpy as np
from torch import manual_seed
from sksurv.util import Surv


def get_cohort():

    host = '/tmp'
    user='postgres'
    passwd='postgres'
    con = psycopg2.connect(dbname ='mimic', user=user, password=passwd, host=host)
    cur = con.cursor()

    # Cohort Table
    cohort_query = 'SELECT * FROM mimiciii.cohort_survival'
    cohort = pd.read_sql_query(cohort_query, con)

    return cohort


def train_test_split(cohort_X, cohort_y):

    ########################################################
    # Cohort
    # ------
    # Total: 9101 admissions with 6379 distinct pacients
    # 4823 pacients with 1 admission -> 52%
    # 1556 pacients with +1 admission -> 48%
    #
    # New cohort
    # ----------
    # Total: 4823 + 2014 + 891 = 7728 admissions
    # patients with 2 admissions -> 2014
    # patients with 3 admissions -> 891
    # 4823/7728 -> 62,4% train
    # 2905/7728 -> 37,6% test
    ########################################################

    cohort = get_cohort()
    cohort_train = cohort.groupby("subject_id").filter(lambda x: len(x) < 2)
    cohort_test = cohort.groupby("subject_id").filter(lambda x: len(x) > 1 and len(x) < 4)

    # id_train = cohort_X.index.intersection(train_index)
    # id_test = cohort_X.index.intersection(test_index)
    # print (id_train, id_test)

    X_train = cohort_X.drop(cohort_test.index)
    X_test = cohort_X.drop(cohort_train.index)
    y_train = cohort_y.drop(cohort_test.index)
    y_test = cohort_y.drop(cohort_train.index)

    return X_train, X_test, y_train, y_test


def train_test_split_nn(seed, size, sa_cohort):

    _ = manual_seed(seed)

    cohort = get_cohort()
    cohort_train = cohort.groupby("subject_id").filter(lambda x: len(x) < 2)
    cohort_test = cohort.groupby("subject_id").filter(lambda x: len(x) > 1 and len(x) < 4)

    test_dataset = sa_cohort.drop(cohort_train.index)
    train_dataset = sa_cohort.drop(cohort_test.index)
    valid_dataset = train_dataset.sample(frac=size)
    train_dataset = train_dataset.drop(valid_dataset.index)

    return train_dataset, valid_dataset, test_dataset


def cox_classical():
    # Get data
    cohort = get_cohort()

    # Binning
    cohort['age_st'] = pd.cut(cohort['age'], np.arange(15, 91, 15))

    # Gender: from categorical to numerical
    cohort.gender.replace(to_replace=dict(F=1, M=0), inplace=True)

    # Change types
    cohort = cohort.astype({'admission_type': 'category', 'ethnicity_grouped': 'category', 'insurance': 'category',
                            'icd_alzheimer': 'category', 'icd_cancer': 'category', 'icd_diabetes': 'category', 'icd_heart': 'category',
                            'icd_transplant': 'category', 'gender': 'category', 'hospital_expire_flag': 'bool',
                            'oasis_score':'category'}, copy=False)

    # Convert categorical variables
    cat = ['gender', 'insurance', 'ethnicity_grouped', 'admission_type', 'oasis_score',
           'icd_alzheimer', 'icd_cancer', 'icd_diabetes', 'icd_heart', 'icd_transplant', 'age_st']
    cohort_df = pd.get_dummies(cohort, columns=cat, drop_first=True)

    # Select features
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age', 'level_0']
    cohort_df.drop(drop, axis=1, inplace=True)

    cohort_X = cohort_df[cohort_df.columns.difference(["los_hospital"])]
    cohort_y = cohort_df["los_hospital"]
    return cohort_X, cohort_y, cohort_df


def cox():
    # Get data
    cohort = get_cohort()

    # Binning
    cohort['age_st'] = pd.cut(cohort['age'], np.arange(15, 91, 15))

    # Gender: from categorical to numerical
    cohort.gender.replace(to_replace=dict(F=1, M=0), inplace=True)

    # Change types
    cohort = cohort.astype({'admission_type': 'category', 'ethnicity_grouped': 'category', 'insurance': 'category',
                            'icd_alzheimer': 'int64', 'icd_cancer': 'int64', 'icd_diabetes': 'int64', 'icd_heart': 'int64',
                            'icd_transplant': 'int64', 'gender': 'int64', 'hospital_expire_flag': 'int64',
                            'oasis_score': 'int64'}, copy=False)

    # Select features
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age', 'level_0']
    cohort.drop(drop, axis=1, inplace=True)

    # Datasets
    cohort_X = cohort[cohort.columns.difference(["los_hospital", "hospital_expire_flag"])]
    cohort_y = cohort[["hospital_expire_flag", "los_hospital"]]
    cohort_y['hospital_expire_flag'] = cohort_y['hospital_expire_flag'].astype(bool)
    cohort_y = Surv.from_dataframe("hospital_expire_flag", "los_hospital", cohort_y)

    return cohort_X, cohort_y


def cox_neural_network():
    # Get data
    cohort = get_cohort()

    # Binning
    cohort['age_st'] = pd.cut(cohort['age'], np.arange(15, 91, 15))

    # Gender: from categorical to numerical
    cohort.gender.replace(to_replace=dict(F=1, M=0), inplace=True)
    cohort = cohort.astype({'admission_type': 'category', 'ethnicity_grouped': 'category', 'insurance': 'category',
                            'icd_alzheimer': 'int64', 'icd_cancer': 'int64', 'icd_diabetes': 'int64', 'icd_heart': 'int64',
                            'icd_transplant': 'int64', 'gender': 'int64', 'hospital_expire_flag': 'int64',
                            'oasis_score': 'int64'}, copy=False)

    # Neural network
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age', 'level_0']
    cohort.drop(drop, axis=1, inplace=True)

    return cohort
