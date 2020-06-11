
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
n = 48
base = 100
base_year = 2015

period = np.arange(n)+1
month_values = np.arange(12)+1

total = np.array([base] + [int(np.ceil(base * (1 + (a/(a - c))*(np.power(1 + (a - c), i) - 1)))) for i in period]).reshape(n+1, 1)
diff = np.diff(total, 1, axis=0)
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

months_col = np.array(months_col).reshape(total[n, 0].item(), 1)
period_col = np.array(period_col).reshape(total[n, 0].item(), 1)


bins = ['18-25', '25-35', '35-45', '45-55', '55-65', '65-80']
age_bins = np.array([0, 1, 2, 3, 4, 5])
weight = np.array([0.1, 0.25, 0.20, 0.20, 0.20, 0.05])
samples_bins = np.random.choice(age_bins, size=total[n, 0], p=weight)
samples = np.array([int(np.random.uniform(int(bins[i].split("-")[0]),
                                 int(bins[i].split("-")[1]))) for i in samples_bins]).reshape(total[n, 0], 1)

data = pd.DataFrame(np.concatenate([months_col, period_col, samples], axis=1), columns=['month', 'period', 'age'])
data['year'] = base_year + (np.ceil(data['period'] / 12) - 1.0).astype(int)
data['date'] = data[['year', 'month']].apply(lambda df : randomdate(df[0], df[1]), axis=1)
data.sort_values(by=['date'], inplace=True)
data.reset_index(drop=True, inplace=True)
data['id'] = data.reset_index()['index']+1

cols = ['id', 'date', 'period', 'month', 'year', 'age']
print(data[cols].head(100))

data[cols].to_csv('data_generation.csv', index=False)



