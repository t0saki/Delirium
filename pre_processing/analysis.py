# Import pandas and matplotlib libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# Read the csv file into a dataframe
df = pd.read_csv("pre_processing/output.csv")

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
    plt.scatter(df[column], df[label])
    # Add labels and title
    plt.xlabel(column)
    plt.ylabel(str(label))
    plt.title(f"Scatterplot of {column} vs {label}")
    # Save the figure to the output folder
    plt.savefig(f"pre_processing/output_scatterplots/{str(count)}.png")
    # Clear the figure for the next plot
    plt.clf()
    count += 1
