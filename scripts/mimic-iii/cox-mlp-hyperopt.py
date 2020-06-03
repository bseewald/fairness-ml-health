import time

import cohort.get_cohort as sa_cohort
import hyperopt_parameters as parameters
import numpy as np
import pandas as pd
import settings
import torch
import torchtuples as tt
from hyperopt import STATUS_OK
from pycox import utils
from pycox.evaluation import EvalSurv
from pycox.models import CoxCC
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn_pandas import DataFrameMapper


def cohort_samples(seed, size, cohort):
    # _ = torch.manual_seed(seed)
    # test_dataset = cohort.sample(frac=size)
    # train_dataset = cohort.drop(test_dataset.index)
    # valid_dataset = train_dataset.sample(frac=size)
    # train_dataset = train_dataset.drop(valid_dataset.index)

    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Feature transforms
    x_train, x_val, x_test = preprocess_input_features(train_dataset, valid_dataset, test_dataset)
    train, val, test = preprocess_target_features(x_train, x_val, x_test,
                                                  train_dataset, valid_dataset, test_dataset)
    return train, val, test


def preprocess_input_features(train_dataset, valid_dataset, test_dataset):
    cols_categorical =  ['insurance', 'ethnicity_grouped', 'age_st', 'oasis_score', 'admission_type']
    categorical = [(col, OrderedCategoricalLong()) for col in cols_categorical]
    x_mapper_long = DataFrameMapper(categorical)

    cols_leave = ['gender',
                  'icd_alzheimer', 'icd_cancer', 'icd_diabetes', 'icd_heart', 'icd_transplant',
                  'first_hosp_stay', 'first_icu_stay']
    leave = [(col, None) for col in cols_leave]
    x_mapper_float = DataFrameMapper(leave)

    x_fit_transform = lambda df: tt.tuplefy(x_mapper_float.fit_transform(df).astype('float32'), x_mapper_long.fit_transform(df))
    x_transform = lambda df: tt.tuplefy(x_mapper_float.transform(df).astype('float32'), x_mapper_long.transform(df))

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


def make_net(train, dropout, num_nodes):
    # Entity embedding
    num_embeddings = train[0][1].max(0) + 1
    embedding_dims = num_embeddings // 2

    in_features = train[0][0].shape[1]
    out_features = 1
    net = tt.practical.MixedInputMLP(in_features,
                                     num_embeddings=num_embeddings,
                                     embedding_dims=embedding_dims,
                                     num_nodes=num_nodes, out_features=out_features,
                                     dropout=dropout, output_bias=False)
    return net


def add_km_censor_modified(ev, durations, events):
    """
        Add censoring estimates obtained by Kaplan-Meier on the test set(durations, 1-events).
    """
    # modified add_km_censor function
    km = utils.kaplan_meier(durations, 1-events)
    surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(durations), axis=1), index=km.index)

    # increasing index (pd.Series(surv.index).is_monotonic)
    surv.drop(0.000000, axis=0, inplace=True)

    return ev.add_censor_est(surv)


def experiment(params):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cohort = sa_cohort.cox_neural_network()
    train, val, test = cohort_samples(seed=settings.seed, size=settings.size, cohort=cohort)

    net = make_net(train, params['dropout'], params['num_nodes'])
    optimizer = tt.optim.AdamWR(decoupled_weight_decay=params['weight_decay'])
    model = CoxCC(net, device=device, optimizer=optimizer, shrink=params['shrink'])
    model.optimizer.set_lr(params['lr'])
    callbacks = [tt.callbacks.EarlyStopping()]
    _ = model.fit(train[0], train[1], batch_size=params['batch'], epochs=settings.epochs, callbacks=callbacks,
                  val_data=val.repeat(10).cat())

    _ = model.compute_baseline_hazards()

    surv = model.predict_surv_df(test[0])

    durations = test[1][0]
    events = test[1][1]

    ev = EvalSurv(surv, durations, events)

    # Setting 'add_km_censor_modified' means that we estimate
    # the censoring distribution by Kaplan-Meier on the test set.
    _ = add_km_censor_modified(ev, durations, events)

    # c-index: the bigger, the better
    cindex = ev.concordance_td()

    # -cindex it's a work around because hyperopt uses fmin function
    return {'loss': -cindex, 'status': STATUS_OK}


def main():

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

    # Open file
    _file = open("files/cox-mlp-hyperopt.txt", "a")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    trials, best = parameters.hyperopt(experiment, "cc")

    # All parameters
    _file.write("All Parameters: \n" + str(trials.trials) + "\n\n")

    # Best Parameters
    _file.write("Best Parameters: " + str(best) + "\n")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()
