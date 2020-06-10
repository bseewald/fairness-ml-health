from time import localtime, strftime

import cohort.get_cohort as sa_cohort
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import settings
import best_parameters
import torch
import torchtuples as tt
from pycox import utils
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from sklearn_pandas import DataFrameMapper


def cohort_samples(seed, size, cohort):

    # Train / valid / test split
    train_dataset, valid_dataset, test_dataset = sa_cohort.train_test_split_nn(seed, size, cohort)

    # Feature transforms
    x_train, x_val, x_test = preprocess_input_features(train_dataset, valid_dataset, test_dataset)
    train, val, test = preprocess_target_features(x_train, x_val, x_test,
                                                  train_dataset, valid_dataset, test_dataset)
    return train, val, test


def preprocess_input_features(train_dataset, valid_dataset, test_dataset):

    cols_categorical = ['insurance', 'ethnicity_grouped', 'age_st', 'oasis_score', 'admission_type']
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


# Training the model
def fit_and_predict(survival_analysis_model, train, val, test,
                    lr, batch, dropout, epoch, weight_decay,
                    num_nodes, device):
    net = make_net(train, dropout, num_nodes)

    optimizer = tt.optim.AdamWR(decoupled_weight_decay=weight_decay)
    model = survival_analysis_model(net, device=device, optimizer=optimizer)
    model.optimizer.set_lr(lr)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(train[0], train[1], batch, epoch, callbacks, val_data=val.repeat(10).cat(), drop_last=True)

    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(test[0])
    surv_v = model.predict_surv_df(val[0])
    return surv, surv_v, model, log


def add_km_censor_modified(ev, durations, events):

    """
        Add censoring estimates obtained by Kaplan-Meier on the test set(durations, 1-events).
    """
    # modified add_km_censor function
    km = utils.kaplan_meier(durations, 1-events)
    surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(durations), axis=1), index=km.index)

    # increasing index
    if pd.Series(surv.index).is_monotonic is False:
        surv.drop(0.000000, axis=0, inplace=True)

    return ev.add_censor_est(surv)


def evaluate(sample, surv):

    durations = sample[1][0]
    events = sample[1][1]

    ev = EvalSurv(surv, durations, events)

    # Setting 'add_km_censor_modified' means that we estimate
    # the censoring distribution by Kaplan-Meier on the test set.
    _ = add_km_censor_modified(ev, durations, events)

    # c-index
    cindex = ev.concordance_td()

    # brier score
    time_grid = np.linspace(durations.min(), durations.max(), 100)
    _ = ev.brier_score(time_grid)
    bscore = ev.integrated_brier_score(time_grid)

    # binomial log-likelihood
    nbll = ev.integrated_nbll(time_grid)

    return cindex, bscore, nbll


def main(seed, index):

    ##################################################################################
    # PyCox Library
    # https://github.com/havakv/pycox
    #
    # CoxPH (DeepServ)
    #
    #     """Cox proportional hazards model parameterized with a neural net.
    #     This is essentially the DeepSurv method [1].
    #
    #     The loss function is not quite the partial log-likelihood, but close.
    #     The difference is that for tied events, we use a random order instead of
    #     including all individuals that had an event at that point in time.
    #
    #     [1] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger.
    #         Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.
    #         BMC Medical Research Methodology, 18(1), 2018.
    #         https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1
    #
    ##################################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cohort = sa_cohort.cox_neural_network()
    train, val, test = cohort_samples(seed=seed, size=settings.size, cohort=cohort)

    # Open file
    _file = open("files/cox-ph/cox-ph.txt", "a")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    best = best_parameters.cox_ph[index]

    surv, surv_v, model, log = fit_and_predict(CoxPH, train, val, test,
                                               lr=best['lr'], batch=best['batch'], dropout=best['dropout'],
                                               epoch=best['epoch'], weight_decay=best['weight_decay'],
                                               num_nodes=best['num_nodes'], device=device)

    fig_time = strftime("%d%m%Y%H%M%S", localtime())

    model.save_net("files/cox-ph/cox-ph-net-" + fig_time + ".pt")
    model.save_model_weights("files/cox-ph/cox-ph-net-weights-" + fig_time + ".txt")
    model.print_weights("files/cox-ph/cox-ph-net-weights-" + fig_time + ".txt")

    # Train, Val Loss
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.grid(True)
    log.plot().get_figure().savefig("img/cox-ph/cox-ph-train-val-loss-" + fig_time + ".png", format="png", bbox_inches="tight")

    # Survival estimates as a dataframe
    estimates = settings.estimates
    plt.ylabel('S(t | x)')
    plt.xlabel('Time')
    surv.iloc[:, :estimates].plot().get_figure().savefig("img/cox-ph/cox-ph-survival-estimates-" + fig_time + ".png", format="png",
                                                         bbox_inches="tight")

    # Evaluate
    cindex_v, bscore_v, bll_v = evaluate(val, surv_v)
    cindex, bscore, bll = evaluate(test, surv)

    # Best Parameters
    _file.write("Best Parameters: " + str(best) + "\n")

    # Scores
    _file.write("Validation \n"
                "C-Index: " + str(cindex_v) + "\n" +
                "Brier Score: " + str(bscore_v) + "\n" +
                "Binomial Log-Likelihood: " + str(bll_v) + "\n")
    _file.write("Test \n"
                "C-Index: " + str(cindex) + "\n" +
                "Brier Score: " + str(bscore) + "\n" +
                "Binomial Log-Likelihood: " + str(bll) + "\n")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    i = 0
    for seed in settings.seed:
        main(seed, i)
        i+=1
