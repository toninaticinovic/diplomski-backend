import pandas as pd

# Read the CSV file
df = pd.read_csv('datasets/classification/haberman.csv')

# Map the last column values from 1 to 0 and from 2 to 1
df.iloc[:, -1] = df.iloc[:, -1].map({1: 0, 2: 1})

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False)