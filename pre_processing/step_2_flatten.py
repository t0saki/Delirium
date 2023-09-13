import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('pre_processing/output.csv')

data = df

# Convert NaN to -max
data_full = pd.DataFrame()
cols = data.columns
for i in cols:
    if data[i].isnull().any():
        data_full[i] = data[i].fillna(0)
    else:
        data_full[i] = data[i]

# Normalize 0-1
scaler = MinMaxScaler()
data_full = pd.DataFrame(scaler.fit_transform(
    data_full), columns=data_full.columns)

# # To dataframe
# data_full = pd.concat(data_full, axis=1)

# Save to csv
data_full.to_csv('pre_processing/output2.csv', index=False)
print('Output file written successfully.')
