import pandas as pd
import psycopg2
from sqlalchemy import create_engine


def normalize_insurance(ins):
    if ins in ['Government', 'Medicaid', 'Medicare']:
        return 'Public'
    elif ins == 'Private':
        return 'Private'
    else:
        return 'Self-Pay'


def hadms_list(icd_list, icu_diagnoses_df):
    hadm_ids_list = set()
    for icd in icd_list:
        patients = icu_diagnoses_df.loc[(icu_diagnoses_df["icd9_code"] == icd)].copy()
        for hadm_id in patients['hadm_id']:
            hadm_ids_list.add(hadm_id)
    return hadm_ids_list


def main():

    #######################
    # POSTGRESQL Connection
    #######################
    host = '/tmp'
    user='postgres'
    passwd='postgres'
    con = psycopg2.connect(dbname ='mimic', user=user, password=passwd, host=host)
    cur = con.cursor()


    ##################
    ## MIMIC-III
    ##################

    # ICD-9 Codes table

    diagnoses_query = '''SELECT * FROM mimiciii.diagnoses_icd;'''
    mimic_diagnoses_df = pd.read_sql_query(diagnoses_query, con)

    # ICD-9 Descriptions table

    diagnoses_descriptions_query = '''SELECT * FROM mimiciii.d_icd_diagnoses;'''
    mimic_diagnoses_descriptions_df = pd.read_sql_query(diagnoses_descriptions_query, con)

    # ICU Stays (patients details)

    icustay_query = 'SELECT * FROM mimiciii.icustay_detail_v2;'
    icustay_details_df = pd.read_sql_query(icustay_query, con)

    # We are not considering MULTI RACE ETHNICITY, NATIVE, UNKNOWN or OTHER
    icustay_details_df = icustay_details_df[(icustay_details_df['ethnicity_grouped'] != 'other') &
                                            (icustay_details_df['ethnicity_grouped'] != 'unknown') &
                                            (icustay_details_df['ethnicity_grouped'] != 'native')]

    # +18 years old (300 years old are patients older than 89)
    icustay_details_df = icustay_details_df[(icustay_details_df['age'] >= 18) & (icustay_details_df['age'] < 300)]

    # insurance
    icustay_details_df['insurance'] = icustay_details_df['insurance'].apply(normalize_insurance)

    # icd9 merge
    icu_diagnoses_df = pd.merge(icustay_details_df, mimic_diagnoses_df, on = ['subject_id', 'hadm_id'], how = 'inner')
    eth_mortality_df = icu_diagnoses_df.groupby(['icd9_code', 'ethnicity_grouped', 'hospital_expire_flag']).size().unstack()
    eth_mortality_df = eth_mortality_df.reset_index()
    eth_mortality_df.columns.names = [None]
    eth_mortality_df.columns = ['icd9_code', 'ethnicity', 'alive', 'dead']
    eth_mortality_df.insert(4, 'total', '0')
    eth_mortality_df = eth_mortality_df.fillna(0)
    eth_mortality_df['total'] = eth_mortality_df['total'].astype(float)
    eth_mortality_df
    # Compute alive, dead and total
    for index, row in eth_mortality_df.iterrows():
        eth_mortality_df.at[index, 'total'] = row['alive'] + row['dead']

    # merge mortality with descriptions from each ICD
    eth_mortality_df = eth_mortality_df.merge(mimic_diagnoses_descriptions_df, left_on='icd9_code', right_on='icd9_code')


    # At this point, we decided to work with 3 types of diseases.
    # Which are among the top 10 causes of death in high-income countries
    # (We are working with a database from a US hospital).
    #
    # Source: https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death

    # 1. Transplanted patients
    transplanted_patients_df = eth_mortality_df[eth_mortality_df['long_title'].str.lower().str.contains('transplant')].copy()

    # 1.1. Mortality significance: only +1 patients dead
    transplanted_patients_df = transplanted_patients_df.loc[transplanted_patients_df['dead'] > 1]

    # 1.2. Remove ICD9 codes with only ONE ETHNICITY
    for index, row in transplanted_patients_df.iterrows():
        rows = transplanted_patients_df.loc[transplanted_patients_df['icd9_code'] == row['icd9_code']]
        if (len(rows) == 1):
            transplanted_patients_df.drop(rows.index, inplace=True)

    transplanted_patients_df = transplanted_patients_df.drop(['row_id', 'short_title'], axis=1)


    # 2. Cancer
    searchfor = ['neoplasm', 'neoplasms', 'sarcoma', 'carcinoma']
    cancer_patients_df = eth_mortality_df[eth_mortality_df['long_title'].str.lower().str.contains('|'.join(searchfor))].copy()

    # 2.1. Mortality significance: only +1 patients dead
    cancer_patients_df = cancer_patients_df.loc[cancer_patients_df['dead'] > 1]

    # 2.2. Remove ICD9 codes with only ONE ETHNICITY
    for index, row in cancer_patients_df.iterrows():
        rows = cancer_patients_df.loc[cancer_patients_df['icd9_code'] == row['icd9_code']]
        if (len(rows) == 1):
            cancer_patients_df.drop(rows.index, inplace=True)

    cancer_patients_df = cancer_patients_df.drop(['row_id', 'short_title'], axis=1)


    # 3. Diabetes
    diabetes_patients_df = eth_mortality_df[eth_mortality_df['long_title'].str.lower().str.contains('diabetes')].copy()

    # 3.1. Mortality significance: only +1 patients dead
    diabetes_patients_df = diabetes_patients_df.loc[diabetes_patients_df['dead'] > 1]

    # 3.2. Remove ICD9 codes with only ONE ETHNICITY
    for index, row in diabetes_patients_df.iterrows():
        rows = diabetes_patients_df.loc[diabetes_patients_df['icd9_code'] == row['icd9_code']]
        if (len(rows) == 1):
            diabetes_patients_df.drop(rows.index, inplace=True)

    diabetes_patients_df = diabetes_patients_df.drop(['row_id', 'short_title'], axis=1)


    # 4. Heart
    searchfor = ['heart', 'myocardial','stroke', 'artery', 'arterial']
    heart_patients_df = eth_mortality_df[eth_mortality_df['long_title'].str.lower().str.contains('|'.join(searchfor))].copy()
    heart_patients_df = heart_patients_df.loc[heart_patients_df['dead'] > 1]

    for index, row in heart_patients_df.iterrows():
        rows = heart_patients_df.loc[heart_patients_df['icd9_code'] == row['icd9_code']]
        if (len(rows) == 1):
            heart_patients_df.drop(rows.index, inplace=True)

    heart_patients_df = heart_patients_df.drop(['row_id', 'short_title'], axis=1)


    # 5. Alzheimer
    alzheimer_patients_df = eth_mortality_df[eth_mortality_df['long_title'].str.lower().str.contains('alzheimer')].copy()

    # 5.1. Mortality significance: only +1 patients dead
    alzheimer_patients_df = alzheimer_patients_df.loc[alzheimer_patients_df['dead'] > 1]

    # 5.2. Remove ICD9 codes with only ONE ETHNICITY
    for index, row in alzheimer_patients_df.iterrows():
        rows = alzheimer_patients_df.loc[alzheimer_patients_df['icd9_code'] == row['icd9_code']]
        if (len(rows) == 1):
            alzheimer_patients_df.drop(rows.index, inplace=True)

    alzheimer_patients_df = alzheimer_patients_df.drop(['row_id', 'short_title'], axis=1)

    # Select admissions (by disease)
    icd9_list_transplants = set(transplanted_patients_df['icd9_code'])
    icd9_list_cancer = set(cancer_patients_df['icd9_code'])
    icd9_list_diabetes = set(diabetes_patients_df['icd9_code'])
    icd9_list_heart = set(heart_patients_df['icd9_code'])
    icd9_list_alzheimer = set(alzheimer_patients_df['icd9_code'])

    hadm_ids_list_transplants = hadms_list(icd9_list_transplants, icu_diagnoses_df)
    hadm_ids_list_cancer = hadms_list(icd9_list_cancer, icu_diagnoses_df)
    hadm_ids_list_diabetes = hadms_list(icd9_list_diabetes, icu_diagnoses_df)
    hadm_ids_list_heart = hadms_list(icd9_list_heart, icu_diagnoses_df)
    hadm_ids_list_alzheimer = hadms_list(icd9_list_alzheimer, icu_diagnoses_df)

    # Cohort Table (final)
    cohort_query = 'SELECT * FROM mimiciii.cohort'
    cohort_df = pd.read_sql_query(cohort_query, con)

    # Preparing for regression
    cohort = cohort_df.copy()
    cohort.dropna(inplace=True)

    cohort = pd.concat([cohort,pd.DataFrame(columns=["icd_alzheimer", "icd_cancer", "icd_diabetes", "icd_heart", "icd_transplant"])])
    cohort.loc[(cohort['hadm_id'].isin(hadm_ids_list_alzheimer)),'icd_alzheimer'] = '1'
    cohort.loc[(cohort['hadm_id'].isin(hadm_ids_list_cancer)),'icd_cancer'] = '1'
    cohort.loc[(cohort['hadm_id'].isin(hadm_ids_list_diabetes)),'icd_diabetes'] = '1'
    cohort.loc[(cohort['hadm_id'].isin(hadm_ids_list_heart)),'icd_heart'] = '1'
    cohort.loc[(cohort['hadm_id'].isin(hadm_ids_list_transplants)), 'icd_transplant']= '1'
    cohort.fillna(value=0, inplace=True)

    # save in Postgres
    print("Saving cohort table...")
    eng = create_engine('postgresql://postgres:postgres@localhost:5432/mimic')
    cohort.to_sql("cohort_survival", con=eng, schema="mimiciii")

if __name__ == "__main__":
    main()