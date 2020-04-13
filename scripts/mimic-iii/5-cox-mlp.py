import pandas as pd
import numpy as np
import psycopg2
import time
from cohort import get_cohort as gh

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


# ---------------------
# Hyperparameter values
# ---------------------
# Layers                           {1, 2, 4}
# Nodes per layer                  {64, 128, 256, 512}
# Dropout                          [0, 0.7]
# Weigh decay                      {0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0} - torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# Batch size                       {64, 128, 256, 512, 1024}
# λ(penalty to the loss function)  {0.1, 0.01, 0.001, 0} - CoxCC(net, optimizer, shrink)

def make_net(train, bn, dpt):
    # Entity embedding
    num_embeddings = train[0][1].max(0) + 1
    embedding_dims = num_embeddings // 2

    in_features = train[0][0].shape[1]
    num_nodes = [32, 32] # 2 layers with 32 nodes each
    out_features = 1
    net = tt.practical.MixedInputMLP(in_features, num_embeddings, embedding_dims,
                                     num_nodes, out_features,
                                     batch_norm=bn, dropout=dpt, output_bias=False)
    return net


# Training the model
def fit_and_predict(survival_analysis_model, train, val, test, lr, bn, dpt, ep, shrink):
    net = make_net(train, bn, dpt)
    model = survival_analysis_model(net, optimizer=tt.optim.Adam, shrink=shrink)
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
    # cohort_y = cohort[["hospital_expire_flag", "los_hospital"]]
    # cohort_X = cohort[cohort.columns.difference(["los_hospital", "hospital_expire_flag"])]

    ##################################
    # PyCox Library
    # https://github.com/havakv/pycox
    #
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
    #
    ##################################

    _ = torch.manual_seed(20)

    # Train / valid / test split
    test_dataset = cohort.sample(frac=0.2)
    train_dataset = cohort.drop(test_dataset.index)
    valid_dataset = train_dataset.sample(frac=0.2)
    train_dataset = train_dataset.drop(valid_dataset.index)

    tt.tuplefy(train_dataset, valid_dataset, test_dataset).lens()

    # Feature transforms
    x_train, x_val, x_test = preprocess_input_features(train_dataset, valid_dataset, test_dataset)
    train, val, test = preprocess_target_features(x_train, x_val, x_test,
                                                  train_dataset, valid_dataset, test_dataset)

    # Open file
    _file = open("files/cox-rsf-v2.txt", "a")

    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")


    surv_cc, model_cc, log_cc = fit_and_predict(CoxCC, train, val, test, lr=0.01, bn=256, dpt=0.1, ep=512)


    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    _file.write("\n*** The last one is the best configuration! ***\n\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()