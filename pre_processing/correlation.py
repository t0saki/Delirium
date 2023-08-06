# # Import pandas library
# import pandas as pd

# # Read the csv file into a dataframe
# df = pd.read_csv("pre_processing/raw_245.csv")

# # Get the column names of the dataframe
# columns = df.columns

# # Get the last column name as the label
# label = columns[-1]

# # Create an empty dictionary to store the correlations
# correlations = {}

# # Loop through the columns except the last one
# for column in columns[:-1]:

#     # Calculate the Spearman correlation between the column and the label
#     correlation = df[column].corr(df[label], method="spearman")

#     # Add the correlation to the dictionary
#     correlations[column] = correlation

# # Convert the dictionary to a dataframe
# correlations_df = pd.DataFrame.from_dict(
#     correlations, orient="index", columns=["Spearman Correlation"])

# # Save the dataframe to a CSV file
# correlations_df.to_csv("pre_processing/correlations.csv")


# Import pandas and scipy libraries
import pandas as pd
from scipy.stats import spearmanr

# Read the csv file into a dataframe
df = pd.read_csv("pre_processing/raw_245.csv")

# Get the column names of the dataframe
columns = df.columns

# Get the last column name as the label
label = columns[-1]

# Create empty lists to store the correlations and p-values
correlations = []
p_values = []

# Loop through the columns except the last one
for column in columns[:-1]:

    # Calculate the Spearman correlation and p-value between the column and the label
    correlation, p_value = spearmanr(df[column], df[label])

    # Add the correlation and p-value to the lists
    correlations.append(correlation)
    p_values.append(p_value)

# Create a dataframe to store the correlations and p-values
correlations_df = pd.DataFrame({'Feature': columns[:-1], 'Spearman Correlation': correlations, 'P-Value': p_values})

# Save the dataframe to a CSV file
correlations_df.to_csv("pre_processing/correlations.csv", index=False)