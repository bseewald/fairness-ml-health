# Survival Analysis
#
# "Survival Analysis is used to estimate the lifespan of a particular population under study.
# It is also called ‘Time to Event’ Analysis as the goal is to estimate the time for an individual or
# a group of individuals to experience an event of interest. This time estimate is the duration
# between birth and death events. Survival Analysis was originally developed and used by Medical
# Researchers and Data Analysts to measure the lifetimes of a certain population."

from time import localtime, strftime

import cohort.get_cohort as cohort
import settings
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit


def main(seed):

    #############################################################
    # Lifelines library
    # https://github.com/CamDavidsonPilon/lifelines
    #
    # Event: hospital_expire_flag (died in hospital or not)
    # Duration: los_hospital (hospital lenght of stay -- in days)
    #############################################################

    # ToDo:
    # Do we censor all individuals that were still under observation at time 30 ?

    # Open file
    _file = open("files/cox.txt", "a")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    cohort_x, cohort_y, cohort_df = cohort.cox_classical()

    # Train / validation / test datasets
    train_size, x_train, x_val, x_test, y_train, y_val, y_test = cohort.train_test_split(seed, settings.size, cohort_x, cohort_y)

    cox = sklearn_adapter(CoxPHFitter, event_col='hospital_expire_flag')
    cx = cox()

    # KFold
    # cv = KFold(n_splits=settings.k, shuffle=True, random_state=seed)

    fold = [-1 for _ in range(train_size)] + [0 for _ in range(x_train.shape[0] - train_size)]
    cv = list(PredefinedSplit(test_fold=fold).split())

    # Training ML model
    gcv = GridSearchCV(cx, {"penalizer": settings._alphas, "l1_ratio": settings._l1_ratios}, cv=cv)

    # Fit
    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))
    gcv_fit = gcv.fit(x_train, y_train)

    # Score
    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))
    gcv_score_val = gcv.score(x_val, y_val)
    gcv_score_test = gcv.score(x_test, y_test)

    # Best Parameters
    _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

    # C-Index
    _file.write("C-Index validation sample: " + str(gcv_score_val) + "\n")
    _file.write("C-Index test sample: " + str(gcv_score_test) + "\n")

    cph = CoxPHFitter(penalizer=gcv_fit.best_params_['penalizer'], l1_ratio=gcv_fit.best_params_['l1_ratio'])
    cph.fit(cohort_df, duration_col="los_hospital", event_col="hospital_expire_flag")

    # Coef
    _file.write("Coeficients:\n" + str(cph.params_) + "\n\n")

    # C-Index score
    cindex = concordance_index(cohort_df['los_hospital'],
                               -cph.predict_partial_hazard(cohort_df),
                               cohort_df['hospital_expire_flag'])
    _file.write("C-Index all dataset: " + str(cindex) + "\n")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    for seed in settings.seed:
        main(seed)
