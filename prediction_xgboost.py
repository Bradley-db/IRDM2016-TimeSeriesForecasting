import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import numpy as np

import pandas as pd
import xgboost as xgb

rmse_list = []

for i in xrange(20):

    # read data with all features
    feature_data = pd.read_csv('./data/feature_data.csv')

    zone_id = i+1

    # train a model for zone_id = 1.
    zone = feature_data[feature_data['zone_id'] == zone_id]


    # split the data to the complete loads data and the miss loads data
    zone_complete = zone[zone['load'].notnull()]
    zone_miss = zone[zone['load'].isnull()]


    # train xgboost with zone_1_complete data
    temp_1_array = np.array(zone_complete)

    # use data before 23/6/2008 to train a model
    # use data between 23/06/2008 to test the model
    X = temp_1_array[:, 2:18]
    X_train = X[:37896, :]
    X_test = X[37896:, :]

    y = temp_1_array[:, 18]
    y_train = y[:37896]
    y_test = y[37896:]

    date1 = datetime.datetime(2008, 6, 23,1)
    date2 = datetime.datetime(2008, 6, 30,6)
    delta = datetime.timedelta(hours=1)
    dates = drange(date1, date2, delta)

    xg_train = xgb.DMatrix(X_train, label=y_train)


    xg_test = xgb.DMatrix(X_test, label=y_test)


    param = {}
    # use softmax multi-class classification
    param['objective'] = 'reg:linear'
    # scale weight of positive examples
    param['eta'] = 0.001
    param['max_depth'] = 5
    param['silent'] = 1
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 99999
    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=150)

    y_pred = bst.predict(xg_test, ntree_limit=bst.best_ntree_limit)


    rmse.append(bst.best_score)

    fig, ax = plt.subplots()
    ax.plot(dates, y_test)
    ax.plot(dates, y_pred)


    ax.set_xlim(dates[0], dates[-1])

    # The hour locator takes the hour or sequence of hours you want to
    # tick, not the base multiple
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 1)))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')
    fig.autofmt_xdate()

    plt.title('Prediction against the true values for zone_%d \n from 6/23/2008 to 6/30/2008' % zone_id)
    plt.legend(('True', 'Prediction'))
    plt.ylabel('load', rotation='horizontal')

    plt.show()

zone = np.arange(1, 21)
rmse_array = np.array(rmse_list)

plt.plot(zone, rmse_array)

plt.title('RMSE of predictions over 20 zones')
plt.xlabel('zone id')
plt.ylabel('Root Mean Squared Error', rotation='horizontal')

plt.show()
