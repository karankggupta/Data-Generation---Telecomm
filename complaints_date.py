import numpy as np
import pandas as pd
import calendar
import random
from datetime import datetime

np.random.seed(10)


def randomdate(year, month, size=10):
    dates = calendar.Calendar().itermonthdates(year, month)
    return np.random.choice([date for date in dates if date.month == month], size=size)


#-------------------------- CREATING DUMMY DATASET -------------------------#

# Randomly creating the sensor readings
date_nov = randomdate(2016, 11)
date_dec = randomdate(2016, 12)
date_jan = randomdate(2017, 1)
date_feb = randomdate(2017, 2)
timestamp = np.concatenate([date_nov, date_dec, date_jan, date_feb], axis=0).reshape((-1, 1))
print(timestamp.dtype)
readings = np.sort(np.random.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], size=timestamp.shape[0])).reshape((-1, 1))
data1 = pd.DataFrame(np.concatenate([timestamp, readings], axis=1), columns=['timestamp', 'readings'])
data1.drop_duplicates(subset=['readings'], inplace=True)
data1['timestamp'] = pd.to_datetime(data1['timestamp'])
print(data1.shape)

# Raw data with hourly records:
date_rng = pd.date_range(start='11/01/2016', end='02/28/2017', freq='H')
raw = pd.DataFrame(date_rng, columns=['date'])
raw1 = raw.copy()
print(raw1.shape)

# -------------------- IMPLEMENTING NON-EQUI JOIN ----------------------------#
# Joining
data1.sort_values(by=['timestamp'], inplace=True)
data2 = data1.shift(periods=-1)
data1 = data1.merge(data2['timestamp'], how='left', right_index=True, left_index=True, suffixes=('_min', '_max'))
data1.loc[pd.isnull(data1['timestamp_max']), 'timestamp_max'] = raw1['date'].max()

# Cartesian Product for implementing Non-Equi join:
data1['key'] = 0
raw1['key'] = 0
raw2 = pd.merge(raw1, data1, how='outer', on='key').query(
    'date >= timestamp_min and date <= timestamp_max')
print(raw2.drop(['key', 'timestamp_min', 'timestamp_max'], axis=1).drop_duplicates(subset=['date'], keep='first'))

# dummy data creation
# n_period = 14
# base_year = np.array([2016] * n_period * 2).reshape((-1, 1))
# period = np.array(np.arange(n_period).tolist() * 2).reshape((-1, 1))
# data = pd.DataFrame(np.concatenate([period, base_year], axis=1), columns=['period', 'year'])
# data['year'] = data['year'] + (data['period'] // 12).astype(int)
# data['month'] = np.mod(data['period'], 12) + 1
# print(data)
#
# data['date'] = data[['year', 'month']].apply(lambda df: randomdate(df[0], df[1]), axis=1)
# print(data.head())


