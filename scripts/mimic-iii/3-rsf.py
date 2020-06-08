from time import localtime, strftime

import cohort.get_cohort as cohort
import numpy as np
import settings
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv


def main(seed):

    #############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # Random Survival Forest
    #
    #############################################################

    old_score = 0

    # Open file
    _file = open("files/cox-rsf.txt", "a")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("########## Init: " + time_string + "\n\n")
    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))

    cohort_X, cohort_y = cohort.cox()

    # Transformation
    Xt = OneHotEncoder().fit_transform(cohort_X)
    Xt = np.column_stack(Xt.values)

    # Train / test split
    X_train, X_test, y_train, y_test = cohort.train_test_split(Xt.transpose(), cohort_y)

    y_train = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_train)
    y_test = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_test)

    # KFold
    cv = KFold(n_splits=settings.k, shuffle=True, random_state=seed)

    # Params
    params = {'n_estimators': settings.n_estimators, 'min_samples_split': settings.split, 'min_samples_leaf': settings.leaf,
              'max_features': settings.max_features, 'n_jobs': settings.n_jobs, 'random_state': [seed]}

    # Train model
    rsf = RandomSurvivalForest()
    gcv = GridSearchCV(rsf, param_grid=params, cv=cv)
    gcv_fit = gcv.fit(X_train, y_train)

    # C-index score
    gcv_score = gcv.score(X_test, y_test)

    _file.write("gcv_score: " + str(gcv_score) + " old_score: " + str(old_score) + "\n")

    # Best Parameters
    _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

    # C-Index
    _file.write("C-Index: " + str(gcv_score) + "\n")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))

    # Close file
    _file.close()


if __name__ == "__main__":
    for seed in settings.seed:
        main(seed)
