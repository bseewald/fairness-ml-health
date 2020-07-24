import psycopg2
import pandas as pd
import numpy as np


def get_cohort():

    host = '/tmp'
    user = 'postgres'
    passwd = 'postgres'
    con = psycopg2.connect(dbname='mimic', user=user, password=passwd, host=host)

    # Cohort Table
    cohort_query = 'SELECT * FROM mimiciii.cohort_survival'
    cohort = pd.read_sql_query(cohort_query, con)
    return cohort


def train_test_split(seed, size, cohort_x, cohort_y):

    ########################################################
    # Cohort
    # ------
    # Total: 9101 admissions with 6379 distinct pacients
    # 4823 pacients with 1 admission -> 52%
    # 1556 pacients with +1 admission -> 48%
    #
    # New cohort
    # ----------
    # Total: 4814 + 2901 = 7715 admissions
    # patients with 2 admissions -> 2014
    # patients with 3 admissions -> 891
    # 4814/7715 -> 62,5% train
    # 2901/7715 -> 37,5% test
    ########################################################

    cohort = get_cohort()
    cohort = cohort.loc[cohort['los_hospital'] > 0]
    cohort_train = cohort.groupby("subject_id").filter(lambda x: len(x) < 2)
    cohort_test = cohort.groupby("subject_id").filter(lambda x: 1 < len(x) < 4)

    # id_train = cohort_x.index.intersection(cohort_train.index)
    # id_test = cohort_x.index.intersection(cohort_test.index)
    # print(id_train, id_test)

    # Duration + Event -> test
    y_test = cohort_y.drop(cohort_train.index)
    y_train_val = cohort_y.drop(cohort_test.index)

    # Features
    x_test = cohort_x.drop(cohort_train.index)
    x_train_val = cohort_x.drop(cohort_test.index)

    cohort_val = cohort_train.sample(frac=size, random_state=seed)
    cohort_train = cohort_train.drop(cohort_val.index)

    x_val = x_train_val.drop(cohort_train.index)
    x_train = x_train_val.drop(x_val.index)

    x_train_val_concat = pd.concat([x_train, x_val])

    # Duration + Event -> train + val (this order is very important!!)
    y_val = y_train_val.drop(cohort_train.index)
    y_train = y_train_val.drop(y_val.index)

    y_train_val_concat = pd.concat([y_train, y_val])

    return len(x_train), x_train_val_concat, x_val, x_test, y_train_val_concat, y_val, y_test


def train_test_split_nn(seed, size, sa_cohort):
    cohort = get_cohort()
    cohort = cohort.loc[cohort['los_hospital'] > 0]
    cohort_train = cohort.groupby("subject_id").filter(lambda x: len(x) < 2)
    cohort_test = cohort.groupby("subject_id").filter(lambda x: 1 < len(x) < 4)

    test_dataset = sa_cohort.drop(cohort_train.index)
    train_dataset = sa_cohort.drop(cohort_test.index)
    valid_dataset = train_dataset.sample(frac=size, random_state=seed)
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
                            'icd_alzheimer': 'category', 'icd_cancer': 'category', 'icd_diabetes': 'category',
                            'icd_heart': 'category', 'icd_transplant': 'category', 'gender': 'category',
                            'hospital_expire_flag': 'bool', 'oasis_score': 'category'}, copy=False)

    # Convert categorical variables
    cat = ['gender', 'insurance', 'ethnicity_grouped', 'admission_type', 'oasis_score',
           'icd_alzheimer', 'icd_cancer', 'icd_diabetes', 'icd_heart', 'icd_transplant', 'age_st']
    cohort_df = pd.get_dummies(cohort, columns=cat, drop_first=True)

    # Select features
    drop = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age', 'level_0']
    cohort_df.drop(drop, axis=1, inplace=True)

    # TODO: there are negative values, investigate why
    cohort_df = cohort_df.loc[cohort_df['los_hospital'] > 0]

    cohort_x = cohort_df[cohort_df.columns.difference(["los_hospital"])]
    cohort_y = cohort_df["los_hospital"]
    return cohort_x, cohort_y, cohort_df


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
    cohort = cohort.loc[cohort['los_hospital'] > 0]

    cohort_x = cohort[cohort.columns.difference(["los_hospital", "hospital_expire_flag"])]
    cohort_y = cohort[["hospital_expire_flag", "los_hospital"]]
    cohort_y['hospital_expire_flag'] = cohort_y['hospital_expire_flag'].astype(bool)
    return cohort_x, cohort_y


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

    cohort = cohort.loc[cohort['los_hospital'] > 0]

    return cohort
