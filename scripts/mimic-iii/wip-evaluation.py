##############
# Evaluation
##############

# Read paper again: how to evaluate!
# RSF: https://nbviewer.jupyter.org/github/sebp/scikit-survival/blob/master/examples/evaluating-survival-models.ipynb

# #### C-Index
#
#     """In survival analysis, the concordance index, or C-index (Harrell Jr et al., 1982), is arguably one of the most commonly applied discriminative evaluation metrics. This is likely a result of its interpretability, as it has a close relationship to classification accuracy (Ishwaran et al., 2008) and ROC AUC (Heagerty and Zheng, 2005). In short, the C-index estimates the probability that, for a random pair of individuals, the predicted survival times of the two individuals have the same ordering as their true survival times. See Ishwaran et al. (2008) for a detailed description [1].
#
#     References:
#     [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
#         Time-to-event prediction with neural networks and Cox regression.
#         Journal of Machine Learning Research, 20(129):1–30, 2019.
#         http://jmlr.org/papers/v20/18-424.html
#     """
#
# #### Brier Score
#
#
#     """The Brier score (BS) for binary classification is a metric of both discrimination and cal-
#     ibration of a model’s estimates. In short, for N binary labels yi ∈ {0,1} with probabil-
#     ities pi of yi = 1, the BS is the mean squared error of the probability estimates pˆi, i.e.,
#     BS = 1/N (yi − pˆi)ˆ2.
#     """
#
#
# We compare the methods using the time-dependent concordance, the integrated Brier score,
# and the integrated binomial log-likelihood. While the concordance solely evaluates a method’s discriminative performance, the Brier score and binomial log-likelihood also evaluate the calibration of the survival estimates.
# We can use the EvalSurv class for evaluation the concordance, brier score and binomial
# log-likelihood. Setting censor_surv='km' means that we estimate the censoring distribution by Kaplan-Meier on the test set.

import pandas as pd
import numpy as np
from pycox.evaluation import EvalSurv

durations = test[1][0]
events = test[1][1]

"""Add censoring estimates obtained by Kaplan-Meier on the test set(durations, 1-events).
"""
from pycox import utils

def add_km_censor_modified(ev):
    # modified add_km_censor function
    km = utils.kaplan_meier(durations, 1-events)
    surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(durations), axis=1), index=km.index)

    # increasing index
    # pd.Series(surv.index).is_monotonic
    surv.drop(0.000000, axis=0, inplace=True)

    return ev.add_censor_est(surv)


# Cox CC
ev_cc = EvalSurv(surv_cc, durations, events)
_ = add_km_censor_modified(ev_cc)
ev_cc.concordance_td()

# deep hit: ev.concordance_td('antolini')

time_grid = np.linspace(durations.min(), durations.max(), 100)
_ = ev_cc.brier_score(time_grid).plot()
ev_cc.integrated_brier_score(time_grid)
ev_cc.integrated_nbll(time_grid)


# Cox Ph
ev_ph = EvalSurv(surv_ph, durations, events)
_ = add_km_censor_modified(ev_ph)
ev_ph.concordance_td()
time_grid = np.linspace(durations.min(), durations.max(), 100)
_ = ev_ph.brier_score(time_grid).plot()
ev_ph.integrated_brier_score(time_grid)
ev_ph.integrated_nbll(time_grid)