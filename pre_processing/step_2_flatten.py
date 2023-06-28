import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('output.csv')

removed_cols = ['Postoperative Olanzapine',
                'Postoperative Fluphenazine', 'Postoperative Flupentixol']
data = df.drop(removed_cols, axis=1)

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
data_full.to_csv('output2.csv', index=False)
print('Output file written successfully.')
