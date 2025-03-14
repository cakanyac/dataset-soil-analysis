import pandas as pd
import numpy as np

df = pd.read_csv("data.csv", sep=";", skipinitialspace=True)
print(df.head())

df.replace(',', '.', regex=True, inplace=True)

cols_to_convert = ['temperature', 'humidity', 'ph', 'rainfall']
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df.info())
print(df.head())

df.to_csv("data_cleaned.csv", index=False, sep=";")
print("Fichier nettoyé sauvegardé sous 'data_cleaned.csv'")
