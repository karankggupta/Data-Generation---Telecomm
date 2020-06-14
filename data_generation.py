
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
c = 0.10
n = 47
base = 60000
base_year = 2015

period = np.arange(n)+1
month_values = np.arange(12)+1

# total = np.array([base] + [int(np.ceil(base * (1 + (a/(a - c))*(np.power(1 + (a - c), i) - 1)))) for i in period]).reshape(n+1, 1)
# diff = np.diff(total, 1, axis=0)
diff = np.zeros(shape=(n, ))
total = np.zeros(shape=(n+1, ))

total[0] = base
diff[0] = a * base

for i in range(1, n):
    total[i] = total[i - 1] + int(total[i - 1] * a) - int(total[i - 1] * c)
    diff[i] = (total[i] * a).astype(int)

diff = diff.reshape(n , 1)
diff = diff.astype(int)
total = base + np.sum(diff)

months = np.pad(month_values, pad_width=(0, n - month_values.shape[0]), mode='wrap').reshape(n, 1)
months_col = np.random.choice(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12]), size=[base, 1], p=[1/12]*12).reshape(base, ).tolist()
period_col = [0] * base
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

data = pd.DataFrame(np.concatenate([months_col, period_col, samples], axis=1), columns=['month', 'period'])
data['year'] = base_year + (np.ceil(data['period'] / 12) - 1.0).astype(int)
data['date'] = data[['year', 'month']].apply(lambda df : randomdate(df[0], df[1]), axis=1)
data.sort_values(by=['date'], inplace=True)
data.reset_index(drop=True, inplace=True)
data['id'] = data.reset_index()['index']+1

cols = ['id', 'date', 'period', 'month', 'year']
print(data[cols].shape)

# data[cols].to_csv('data_generation.csv', index=False)



