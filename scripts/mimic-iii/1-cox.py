# Survival Analysis
#
# "Survival Analysis is used to estimate the lifespan of a particular population under study.
# It is also called ‘Time to Event’ Analysis as the goal is to estimate the time for an individual or
# a group of individuals to experience an event of interest. This time estimate is the duration
# between birth and death events. Survival Analysis was originally developed and used by Medical
# Researchers and Data Analysts to measure the lifetimes of a certain population."

import time

import numpy as np
import pandas as pd
import cohort.get_cohort as cohort
import settings
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split


# TODO: implement!
def brier_score():
    return None


def binomial_log_likelihood():
    return None


def main():

    #############################################################
    # Lifelines library
    # https://github.com/CamDavidsonPilon/lifelines
    #
    # Event: hospital_expire_flag (died in hospital or not)
    # Duration: los_hospital (hospital lenght of stay -- in days)
    #############################################################

    # To-Do:
    # Do we censor all individuals that were still under observation at time 30 ?

    # Open file
    _file = open("files/cox.txt", "a")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    cohort_X, cohort_y, cohort_df = cohort.cox_classical()

    # Train / test samples
    X_train, X_test, y_train, y_test = train_test_split(cohort_X, cohort_y)
    # X_train, X_test, y_train, y_test = train_test_split(cohort_X, cohort_y, test_size=settings.size, random_state=settings.seed)

    cox = sklearn_adapter(CoxPHFitter, event_col='hospital_expire_flag')
    cx = cox()

    # KFold
    cv = KFold(n_splits=settings.k, shuffle=True, random_state=settings.seed)

    # Training ML model
    gcv = GridSearchCV(cx, {"penalizer": settings._alphas, "l1_ratio": settings._l1_ratios}, cv=cv)

    # Fit
    print(time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime()))
    gcv_fit = gcv.fit(X_train, y_train)

    # Score
    print(time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime()))
    gcv_score = gcv.score(X_test, y_test)

    # Best Parameters
    _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

    # C-Index
    _file.write("C-Index test sample: " + str(gcv_score) + "\n")

    cph = CoxPHFitter(penalizer=gcv_fit.best_params_['penalizer'], l1_ratio=gcv_fit.best_params_['l1_ratio'])
    cph.fit(cohort_df, duration_col="los_hospital", event_col="hospital_expire_flag")

    # Coef
    _file.write("Coeficients:\n" + str(cph.params_) + "\n\n")

    # C-Index score
    cindex = concordance_index(cohort_df['los_hospital'],
                               -cph.predict_partial_hazard(cohort_df),
                               cohort_df['hospital_expire_flag'])
    _file.write("C-Index all dataset: " + str(cindex) + "\n")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()


#################################
# FAIRNESS AND SURVIVAL ANALYSIS
#################################

# group fairness OK
# P(S > sHR | G = m) = P(S > sHR | G = f)

# group fairness NOK
# P(S > s | G = asian) = P(S > s | G = not asian)

# Conditional Statistical Parity
# P(S > s | L1 = l1, L2 = l2, E = black) = P(S > s | L1 = l1, L2 = l2, E = not black)
