import seaborn as sns
import matplotlib.pyplot as plt
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

for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel(col)
    plt.ylabel('Fréquence')
    plt.title(f'Distribution de {col}')
    plt.grid(True)
    plt.show()

df_numeric = df.select_dtypes(include=['number'])
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de Corrélation")
plt.show()
