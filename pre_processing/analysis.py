# Import pandas and matplotlib libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# Read the csv file into a dataframe
df = pd.read_csv("pre_processing/raw_245.csv")

# Get the column names of the dataframe
columns = df.columns

# Get the last column name as the label
label = columns[-1]

# Create a folder to store the output scatterplots
if not os.path.exists("output_scatterplots"):
    os.mkdir("output_scatterplots")

count = 0

# Loop through the columns except the last one
for column in columns[:-1]:

    # Create a scatterplot of the column vs the label
    plt.scatter(df[column], df[label], alpha=0.5)

    # Add labels and title
    plt.xlabel(column)
    plt.ylabel(str(label))
    plt.title(f"Scatterplot of {column} vs {label}")

    # Get the unique values in the column
    unique_values = df[column].unique()

    # Loop through the unique values
    for value in unique_values:

        # Get the indices of the rows with the current value
        indices = df[column] == value

        # Get the number of rows with the current value
        num_rows = sum(indices)

        # Set the size of the dots based on the number of rows with the current value
        size = 50 + 100 * num_rows / len(df)

        # Create a scatterplot of the rows with the current value
        plt.scatter(df[column][indices], df[label][indices], s=size)

    # Save the figure to the output folder
    plt.savefig(f"pre_processing/output_scatterplots/{str(count)}.png")

    # Clear the figure for the next plot
    plt.clf()

    count += 1