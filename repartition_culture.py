import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Entraînement du modèle...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

print("Rapport de classification :")
print(classification_report(y_test, y_pred))

joblib.dump(model, "trained_model.pkl")
print("Modèle entraîné sauvegardé sous 'trained_model.pkl'")

df_cleaned = pd.read_csv("data_cleaned.csv", sep=";", skipinitialspace=True)

if 'label' in df_cleaned.columns:
    label_counts = df_cleaned['label'].value_counts()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.xlabel("Type de culture")
    plt.ylabel("Nombre d'occurrences")
    plt.title("Répartition des cultures dans le dataset")
    plt.xticks(rotation=90)
    plt.show()
else:
    print("Erreur : La colonne 'label' n'existe pas dans le fichier nettoyé.")
