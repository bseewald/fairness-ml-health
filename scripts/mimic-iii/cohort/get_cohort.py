import psycopg2
import pandas as pd

#######################
# POSTGRESQL Connection
#######################

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