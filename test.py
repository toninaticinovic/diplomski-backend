import pandas as pd

# Read the CSV file
df = pd.read_csv('datasets/regression/bike_rent.csv')

weather_mapping = {
    1: 'Clear',
    2: 'Cloudy',
    3: 'Light Precipitation',
    4: 'Heavy Precipitation'
}

df['weathersit'] = df['weathersit'].map(weather_mapping)


# Save the modified DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False)