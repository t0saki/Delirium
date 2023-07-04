import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('datasets/20220328-or-eng-shrink.csv')

removed_cols = ['Postoperative Olanzapine',
                'Postoperative Fluphenazine', 'Postoperative Flupentixol']
data = df.drop(removed_cols, axis=1)

# Convert time string to timestamp
time_cols1 = ['Surgery Date', 'Surgery Start Time',
              'Surgery End Time']  # 2019-11-25 17:54:00
time_cols2 = ['Admission Time', 'Discharge Time']  # 2019/11/21 9:14
for col in time_cols1:
    # To timestamp
    data[col] = pd.to_datetime(data[col], format='%Y-%m-%d %H:%M:%S')
for col in time_cols2:
    data[col] = pd.to_datetime(data[col], format='%Y/%m/%d %H:%M')

# Convert datetime to float
for col in time_cols1:
    data[col] = data[col].astype('int64') / 10**9
for col in time_cols2:
    data[col] = data[col].astype('int64') / 10**9

# # Add label column to each column to indicate NaN values
# data_full = pd.DataFrame()
# cols = data.columns
# for i in cols:
#     # if have NaN
#     if data[i].isnull().any():
#         data_full[i] = data[i]
#         # if data_full[i] has NaN, set to 0
#         for j in data_full[i].index:
#             if pd.isnull(data_full[i][j]):
#                 data_full[i][j] = 0
#         data_full[i+'_label'] = data[i].isnull().astype(int)
#     else:
#         data_full[i] = data[i]

# Convert NaN to -max
data_full = pd.DataFrame()
cols = data.columns
for i in cols:
    # if have NaN
    if data[i].isnull().any():
        data_full[i] = data[i].fillna(-data[i].max())
    else:
        data_full[i] = data[i]

# Normalize 0-1
scaler = MinMaxScaler()
data_full = pd.DataFrame(scaler.fit_transform(
    data_full), columns=data_full.columns)

# # To dataframe
# data_full = pd.concat(data_full, axis=1)

# Save to csv
data_full.to_csv('datasets/20220328-or-eng-shrink-full.csv', index=False)
