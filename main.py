import pandas as pd

# Load CSV file
df = pd.read_csv('data-final.csv', sep='\t')

# define worth to keep columns
columns_to_keep = [col for col in df.columns if
                   not col.endswith('_E') and
                   col not in ['dateload', 'screenw', 'screenh', 'introelapse', 'testelapse',
                               'endelapse', 'IPC', 'country', 'lat_appx_lots_of_err', 'long_appx_lots_of_err']]

# Elimina randuri
df_cleaned = df.dropna(subset=columns_to_keep)

df_cleaned = df_cleaned[columns_to_keep]

# Convert back to int8
df_cleaned = df_cleaned.astype('int')

# Override original dataset
df_cleaned.to_csv('data-final.csv', sep='\t', index=False)
