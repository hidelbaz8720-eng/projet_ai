import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\pc\Documents\IA cours\projet\export_2023-12-31-23-00_to_2026-11-14-22-59.csv"
)

# %%
df.head()

# %%
df.tail()

# %%
print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])
print("Dimensions :", df.shape)

# %%
print(df.columns.tolist())

# %%
df.info()

# %%
df.describe().T

# %%
missing = pd.DataFrame({
    'Missing Values': df.isnull().sum(),
    'Percentage (%)': (df.isnull().sum() / len(df)) * 100
})

missing

# %%

# Nombre d'observations pour chaque état
status_counts = df["WateringStatus"].value_counts()

print(status_counts)



# %%
# Pourcentage de chaque état
status_percent = df["WateringStatus"].value_counts(
    normalize=True
) * 100

print("\nPourcentages :")
print(status_percent.round(2))


# %%

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="WateringStatus",
    hue="WateringStatus",
    legend=False
)

plt.title("Répartition de WateringStatus")
plt.xlabel("État de l'arrosage")
plt.ylabel("Nombre d'observations")

plt.show()


# %%

# Variables numériques à étudier
variables = [
    "temp",
    "hum",
    "moistureLevel",
    "totalLitres",
    "pres",
    "waterFlow"
]

# Création des histogrammes
df[variables].hist(
    figsize=(15, 10),
    bins=40,
    edgecolor="black"
)

plt.suptitle(
    "Distribution des variables numériques",
    fontsize=16
)

plt.tight_layout()
plt.show()


# %%


# %%

# Calcul de la corrélation
correlation = df[variables].corr()

# Visualisation
plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Matrice de corrélation")
plt.show()


# %%

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce",
    utc=True
)

print(df["datetime"].head())


# %%

df_time = df.sort_values("datetime")


# %%

# Échantillonnage pour alléger le graphique
sample = df_time.iloc[::100]

plt.figure(figsize=(15, 5))

plt.plot(
    sample["datetime"],
    sample["temp"]
)

plt.title("Évolution de la température")
plt.xlabel("Date")
plt.ylabel("Température")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# %%
#

# %%

