import matplotlib as plt
import seaborn as sns


def plot_error_by_feature(df, model, feature=None):

    if feature:
        censored_subjects = df.loc[(~df['hospital_expire_flag'].astype(bool)) & (df[feature])]
    else:
        censored_subjects = df.loc[~df['hospital_expire_flag'].astype(bool)]

    y = censored_subjects['los_hospital']
    y_hat = model.predict_median(censored_subjects, conditional_after=y).values[:, 0]
    error = y_hat - y

    # plot
    sns.kdeplot(error, label=feature)
    plt.title('Error')
    plt.xlabel('Time in days')
    plt.ylabel('Density')


# if __name__ == "__main__":
#     plot_error_by_feature()
#     plot_error_by_feature('gender_M')
#     plot_error_by_feature('ethnicity_grouped_black')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
#                borderaxespad=0., labels=('all cohort', 'by gender', 'black'))
