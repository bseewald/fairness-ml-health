from time import localtime, strftime

import cohort.get_cohort as cohort
import matplotlib.pyplot as plt
import pandas as pd
import settings
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv


def main(seed):

    ############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # CoxnetSurvivalAnalysis
    #
    #     """Cox's proportional hazard's model with elastic net penalty.
    #
    #     References
    #     ----------
    #     .. [1] Simon N, Friedman J, Hastie T, Tibshirani R.
    #            Regularization paths for Cox’s proportional hazards model via coordinate descent.
    #            Journal of statistical software. 2011 Mar;39(5):1.
    #
    ############################################################

    old_score = 0

    # Open file
    _file = open("files/cox-net.txt", "a")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    # Cohort
    cohort_x, cohort_y = cohort.cox()

    # OneHot
    cohort_x_train = OneHotEncoder().fit_transform(cohort_x)

    # Train / validation / test datasets
    train_size, x_train, x_val, x_test, y_train, y_val, y_test = cohort.train_test_split(seed, settings.size, cohort_x_train, cohort_y)

    cohort_y_train = Surv.from_dataframe("hospital_expire_flag", "los_hospital", cohort_y)
    y_train = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_train)
    y_val = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_val)
    y_test = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_test)

    # KFold
    # cv = KFold(n_splits=settings.k, shuffle=True, random_state=seed)

    fold = [-1 for _ in range(train_size)] + [0 for _ in range(x_train.shape[0] - train_size)]
    cv = PredefinedSplit(test_fold=fold)

    _file.write("Tuning hyper-parameters\n\n")
    _file.write("Alphas: " + str(settings._alphas_cn) + "\n\n")

    # Training Model
    for ratio in settings._l1_ratios_cn:
        coxnet = CoxnetSurvivalAnalysis(alphas=settings._alphas_cn, l1_ratio=ratio, fit_baseline_model=True).fit(cohort_x_train, cohort_y_train)

        _file.write("\nL1 Ratio: " + str(ratio) + "\n")

        # GridSearchCV
        gcv = GridSearchCV(coxnet, {"alphas": [[v] for v in coxnet.alphas_]}, cv=cv)

        # Fit
        gcv_fit = gcv.fit(x_train, y_train)

        # Score
        gcv_score_val = gcv.score(x_val, y_val)
        gcv_score_test = gcv.score(x_test, y_test)

        _file.write("gcv_score: " + str(gcv_score_val) + " old_score: " + str(old_score) + "\n")
        if gcv_score_val > old_score:

            old_score = gcv_score_val

            # Results
            results = pd.DataFrame(gcv_fit.cv_results_)

            alphas = results.param_alphas.map(lambda x: x[0])
            mean = results.mean_test_score
            std = results.std_test_score

            # Plot
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(alphas, mean)
            ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
            ax.set_xscale("log")
            ax.set_ylabel("c-index")
            ax.set_xlabel("alpha")
            ax.axvline(gcv.best_params_['alphas'][0], c='C1')
            ax.grid(True)
            fig.savefig("img/coxnet.png", format="png", bbox_inches="tight")

            # Best Parameters
            _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

            # C-Index
            _file.write("C-Index Validation: " + str(gcv_score_val) + "\n")
            _file.write("C-Index Test: " + str(gcv_score_test) + "\n")

            # Coef
            coef = pd.Series(gcv_fit.best_estimator_.coef_[:, 0], index=cohort_x_train.columns)
            _file.write("Coeficients:\n" + str(coef[coef != 0]) + "\n\n")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    _file.write("\n*** The last one is the best configuration! ***\n\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    for seed in settings.seed:
        main(seed)
