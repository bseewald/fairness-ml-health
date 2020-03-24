import pandas as pd
import numpy as np
import psycopg2
import time
from time import gmtime, strftime

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from pycox.models import CoxCC
from pycox.evaluation import EvalSurv


def preprocess_input_features(train_dataset, valid_dataset, test_dataset):
    cols_categorical =  ['insurance', 'ethnicity_grouped', 'age_st',
                         'oasis_score', 'admission_type']
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]
    x_mapper_long = DataFrameMapper(categorical)

    cols_leave = ['gender',
                  'icd_alzheimer', 'icd_cancer', 'icd_diabetes', 'icd_heart', 'icd_transplant',
                  'first_hosp_stay', 'first_icu_stay']
    leave = [(col, None) for col in cols_leave]
    x_mapper_float = DataFrameMapper(leave)

    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df).astype('float32'),
                                            x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df).astype('float32'),
                                        x_mapper_long.transform(df))

    x_train = x_fit_transform(train_dataset)
    x_val = x_transform(valid_dataset)
    x_test = x_transform(test_dataset)

    return x_train, x_val, x_test


def preprocess_target_features(x_train, x_val, x_test,
                               train_dataset, valid_dataset, test_dataset):

    get_target = lambda df: (df['los_hospital'].values.astype('float32'),
                             df['hospital_expire_flag'].values.astype('float32'))

    y_train = get_target(train_dataset)
    y_val = get_target(valid_dataset)
    y_test = get_target(test_dataset)

    train = tt.tuplefy(x_train, y_train)
    val = tt.tuplefy(x_val, y_val)
    test = tt.tuplefy(x_test, y_test)

    return train, val, test


def make_net(train, bn, dpt):
    # Entity embedding
    num_embeddings = train[0][1].max(0) + 1
    embedding_dims = num_embeddings // 2

    in_features = train[0][0].shape[1]
    num_nodes = [32, 32]
    out_features = 1
    net = tt.practical.MixedInputMLP(in_features, num_embeddings, embedding_dims,
                                    num_nodes, out_features,
                                    batch_norm=bn, dropout=dpt, output_bias=False)
    return net


# Training the model
def fit_and_predict(survival_analysis_model, train, val, test, lr, bn, dpt, ep):
    net = make_net(train, bn, dpt)
    model = survival_analysis_model(net, optimizer=tt.optim.Adam)
    model.optimizer.set_lr(lr)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(train[0], train[1], bn, ep, callbacks, val_data=val.repeat(10).cat())

    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(test[0])
    return surv, model, log


# Finding best learning rate
def best_lr(model, train, batch_size):
    lrfinder = model.lr_finder(train[0], train[1], batch_size, tolerance=10)
    return lrfinder.get_best_lr()


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
    drop_nn = ['index', 'subject_id', 'hadm_id', 'icustay_id', 'dod', 'admittime', 'dischtime', 'ethnicity', 'hospstay_seq',
            'intime', 'outtime', 'los_icu', 'icustay_seq', 'row_id', 'seq_num', 'icd9_code', 'age']
    cohort_nn = cohort.drop(drop_nn, axis=1)

    # Gender: from categorical to numerical
    cohort_nn.gender.replace(to_replace=dict(F=1, M=0), inplace=True)

    ##################################
    # PyCox Library
    # https://github.com/havakv/pycox
    ##################################

    np.random.seed(1234)
    _ = torch.manual_seed(123)

    # Train / valid / test split
    test_dataset = cohort_nn.sample(frac=0.2)
    train_dataset = cohort_nn.drop(test_dataset.index)
    valid_dataset = train_dataset.sample(frac=0.2)
    train_dataset = train_dataset.drop(valid_dataset.index)

    tt.tuplefy(train_dataset, valid_dataset, test_dataset).lens()

    # Feature transforms
    x_train, x_val, x_test = preprocess_input_features(train_dataset, valid_dataset, test_dataset)
    train, val, test = preprocess_target_features(x_train, x_val, x_test,
                                                  train_dataset, valid_dataset, test_dataset)

    # Cox-MLP (CC)
    #
    #     """Cox proportional hazards model parameterized with a neural net and
    #     trained with case-control sampling [1].
    #     This is similar to DeepSurv, but use an approximation of the loss function.
    #
    #     References:
    #     [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
    #         Time-to-event prediction with neural networks and Cox regression.
    #         Journal of Machine Learning Research, 20(129):1–30, 2019.
    #         http://jmlr.org/papers/v20/18-424.html
    #     """

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("init: " + time_string)

    surv_cc, model_cc, log_cc = fit_and_predict(CoxCC, train, val, test, lr=0.01, bn=256, dpt=0.1, ep=512)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("final: " + time_string)

    # TO-DO: ML!


if __name__ == "__main__":
    main()