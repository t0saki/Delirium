import pandas as pd
from googletrans import Translator

# Read the Excel file into a pandas dataframe
df = pd.read_excel("raw_245.xlsx")

# Translate the column names from Chinese to English
translator = Translator()
df.columns = [translator.translate(col, dest='en').text for col in df.columns]

# Convert the dataframe to CSV format and save it to a file
df.to_csv("raw_245.csv", index=False)