from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import os


def generate_training_samples(deer_list, lag=20):

    Y1_total = []
    Y2_total = []
    X1_total = []
    X2_total = []

    Y1_mu_total = []
    Y2_mu_total = []
    Y1_sigma_total = []
    Y2_sigma_total = []

    for num in deer_list:
        df = pd.read_csv('Female/Processed/' + str(num) + '.csv')

        df['eval'] = 0
        df['X_true'] = df['X']
        df['Y_true'] = df['Y']
        # set 20% of data to be missing as test data
        rng = np.random.RandomState(42)
        missing_ratio = 0.5
        time_points_to_eval = rng.choice(len(df), int(missing_ratio * len(df)), replace=False)
        df.loc[time_points_to_eval, 'eval'] = 1
        df.loc[time_points_to_eval, 'X'] = np.nan
        df.loc[time_points_to_eval, 'Y'] = np.nan
        df.loc[time_points_to_eval, 'covariate'] = 0


        df['diff_jul'] = df['jul'].diff()
        df.loc[0, 'diff_jul'] = 0


        Y1_list = []
        Y2_list = []
        X1_list = []
        X2_list = []

        Y1_mu_list = []
        Y2_mu_list = []
        Y1_sigma_list = []
        Y2_sigma_list = []



        for i in range(lag, len(df)):
            if df.loc[i, 'eval'] == 0:
                continue
            cur = df.loc[i-lag:i, :]
            cur = cur.reset_index(drop=True)
            # min-max normalization of X and Y
            Y1_mu = cur.loc[0:lag-1, 'X'].min()
            Y2_mu = cur.loc[0:lag-1, 'Y'].min()
            Y1_sigma = cur.loc[0:lag-1, 'X'].max() - Y1_mu
            Y2_sigma = cur.loc[0:lag-1, 'Y'].max() - Y2_mu
            cur.loc[0:lag-1, 'X'] = (cur['X'] - Y1_mu) / Y1_sigma
            cur.loc[0:lag-1, 'Y'] = (cur['Y'] - Y2_mu) / Y2_sigma

            # impute missing values in cur
            cur['X'] = cur['X'].fillna(0)
            cur['Y'] = cur['Y'].fillna(0)

            # missing values in covariate are set to 0
            cur['covariate'] = cur['covariate'].fillna(0)


            Y1 = (cur.loc[lag, 'X_true'] - Y1_mu) / Y1_sigma
            Y2 = (cur.loc[lag, 'Y_true'] - Y2_mu) / Y2_sigma


            # currrent location as prediction target
            Y1_list.append(Y1)
            Y2_list.append(Y2)

            Y1_mu_list.append(Y1_mu)
            Y2_mu_list.append(Y2_mu)
            Y1_sigma_list.append(Y1_sigma)
            Y2_sigma_list.append(Y2_sigma)

            # lagged locations as features
            lagged_Y1 = cur.loc[0:lag-1, 'X'].values
            lagged_Y2 = cur.loc[0:lag-1, 'Y'].values

            # time intervals as features
            time_intervals = cur.loc[0:lag-1, 'diff_jul'].values

            # covariates as features
            covariates = cur.loc[0:lag-1, ['month', 'day', 'hour', 'covariate']].values

            # concatenate all features
            X1 = np.concatenate([lagged_Y1, lagged_Y2, time_intervals, covariates.flatten()])
            X2 = np.concatenate([lagged_Y1, lagged_Y2, time_intervals, covariates.flatten()])

            X1_list.append(X1)
            X2_list.append(X2)

        Y1_list = np.array(Y1_list)
        Y2_list = np.array(Y2_list)
        X1_list = np.array(X1_list)
        X2_list = np.array(X2_list)

        Y1_total.append(Y1_list)
        Y2_total.append(Y2_list)
        X1_total.append(X1_list)
        X2_total.append(X2_list)

        Y1_mu_total.append(Y1_mu_list)
        Y2_mu_total.append(Y2_mu_list)
        Y1_sigma_total.append(Y1_sigma_list)
        Y2_sigma_total.append(Y2_sigma_list)


    Y1_total = np.concatenate(Y1_total, axis=0)
    Y2_total = np.concatenate(Y2_total, axis=0)
    X1_total = np.concatenate(X1_total, axis=0)
    X2_total = np.concatenate(X2_total, axis=0)

    Y1_mu_total = np.concatenate(Y1_mu_total, axis=0)
    Y2_mu_total = np.concatenate(Y2_mu_total, axis=0)
    Y1_sigma_total = np.concatenate(Y1_sigma_total, axis=0)
    Y2_sigma_total = np.concatenate(Y2_sigma_total, axis=0)




    return Y1_total, Y2_total, X1_total, X2_total, Y1_mu_total, Y2_mu_total, Y1_sigma_total, Y2_sigma_total






if __name__ == '__main__':
    deer_id_list = sorted([int(f.split('.')[0][-4:]) for f in os.listdir('Female/TagData') if f.endswith('.csv')])
    # randomly select 20% of the deer ids as testing data
    rng = np.random.RandomState(42)
    rng.shuffle(deer_id_list)

    # training data
    Y1_train, Y2_train, X1_train, X2_train, Y1_mu_total, Y2_mu_total, Y1_sigma_total, Y2_sigma_total = generate_training_samples(deer_id_list[:int(0.8 * len(deer_id_list))][:100])
    print('shape of Y1_train:', Y1_train.shape)
    print('shape of X1_train:', X1_train.shape)
    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2 = RandomForestRegressor(n_estimators=100, random_state=42)
    model1.fit(X1_train, Y1_train)
    model2.fit(X2_train, Y2_train)

    # testing data
    Y_mae_list = []
    n_obs_list = []
    print('testing data:')
    print(deer_id_list[int(0.8 * len(deer_id_list)):])
    for num in deer_id_list[int(0.8 * len(deer_id_list)):]:
        # print('deer_id:', num)

        Y1_test, Y2_test, X1_test, X2_test, Y1_mu, Y2_mu, Y1_sigma, Y2_sigma = generate_training_samples([num])
        n_obs = Y1_test.shape[0]

        Y1_pred = model1.predict(X1_test)
        Y2_pred = model2.predict(X2_test)

        Y1_mae = np.mean(np.abs((Y1_pred - Y1_test) * Y1_sigma))
        Y2_mae = np.mean(np.abs((Y2_pred - Y2_test) * Y2_sigma))
        Y_mae = (Y1_mae + Y2_mae) / 2

        Y_mae_list.append(Y_mae)
        n_obs_list.append(n_obs)

    Y_mae_list = np.array(Y_mae_list)
    n_obs_list = np.array(n_obs_list)
    Y_mae = np.sum(Y_mae_list * n_obs_list) / n_obs_list.sum()
    print('mean mae:', Y_mae)