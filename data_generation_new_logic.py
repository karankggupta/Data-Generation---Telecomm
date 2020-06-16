
# start_year = 2015
# end_year = 2018

# total = int(np.ceil(base * (1 + (a/(a - c))*(np.power(1 + (a - c), n) - 1))))
# print("acquisition : ", a)
# print("churn : ", c)
# print("periods : ", n)
# print("base : ", base)
# print("total : ", total)


import numpy as np
import pandas as pd
import calendar, random

def randomdate(year, month):
    dates = calendar.Calendar().itermonthdates(year, month)
    return random.choice([date for date in dates if date.month == month])

np.random.seed(10)
a = 0.15
c = 0.02
n = 48
base = 100
base_year = 2016

period = np.arange(n)+1
month_values = np.arange(12)+1

# total = np.array([base] + [int(np.ceil(base * (1 + (a/(a - c))*(np.power(1 + (a - c), i) - 1)))) for i in period]).reshape(n+1, 1)
# diff = np.diff(total, 1, axis=0)
diff = np.zeros(shape=(n, ))
total = np.zeros(shape=(n, ))
churn = np.zeros(shape=(n, ))
acq = np.zeros(shape=(n, ))

total[0] = base
diff[0] = base
acq[0] = 0
churn[0] = 2
# diff[0] = base

for i in range(1, n):
    acq[i] = np.round(total[i - 1] * a + 1e-6)
    total[i] = total[i - 1] + acq[i] - churn[i - 1]
    churn[i] = np.round(total[i] * c + 1e-6)
    diff[i] = acq[i]

print(pd.DataFrame(np.concatenate([total.reshape(n,1), acq.reshape(n,1), churn.reshape(n,1)], axis=1), columns=['total', 'acq', 'churn']))

diff = diff.reshape(n , 1)
diff = diff.astype(int)
total = np.sum(diff)

months = np.pad(month_values, pad_width=(0, n - month_values.shape[0]), mode='wrap').reshape(n, 1)
# months_col = np.random.choice(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12]), size=[base, 1], p=[1/12]*12).reshape(base, ).tolist()
months_col = []
period_col = []
counter = 0

for month in months:
    months_col.extend([month.item()] * diff[counter, 0].item())
    counter += 1

counter = 0
for periods in period:
    period_col.extend([periods.item()] * diff[counter, 0].item())
    counter += 1

months_col = np.array(months_col).reshape(total, 1)
period_col = np.array(period_col).reshape(total, 1)

data = pd.DataFrame(np.concatenate([months_col, period_col], axis=1), columns=['month', 'period'])
data['year'] = base_year + (np.ceil(data['period'] / 12) - 1.0).astype(int)
data['date'] = data[['year', 'month']].apply(lambda df : randomdate(df[0], df[1]), axis=1)
data.sort_values(by=['date'], inplace=True)
data.reset_index(drop=True, inplace=True)
data['id'] = data.reset_index()['index']+1

cols = ['id', 'date', 'period', 'month', 'year']
data[cols].to_csv('teleco_data.csv',index=False)

# data[cols].to_csv('data_generation.csv', index=False)
