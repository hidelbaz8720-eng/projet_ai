# # 03 – Feature Engineering
# ## Projet : Smart City Environment – Smart Irrigation and Water Management
#
# **Objectif :** préparer les variables utiles aux quatre volets du projet :
# 1. Classification : prédiction de `WateringStatus`.
# 2. Régression : estimation de `totalLitres`.
# 3. Détection d'anomalies : analyse des mesures des capteurs.
# 4. Clustering : découverte de profils environnementaux similaires.
#
# > Ce notebook commence à partir du CSV déjà nettoyé. Il ne refait pas le Data Cleaning. Les valeurs manquantes des variables explicatives sont conservées à cette étape : leur imputation et la standardisation seront réalisées dans les pipelines des modèles, pour éviter les fuites de données.
#

# %%
# ## 1. Importer les bibliothèques et charger le dataset
#
# Adapte `INPUT_CSV` si ton fichier nettoyé se trouve dans un autre dossier ou porte un autre nom.

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# Chemin du dataset nettoyé : à adapter selon ton organisation
INPUT_CSV = Path("../dataset/cleaned.csv")

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_CSV.resolve()}\n"
        "Modifie INPUT_CSV pour indiquer le chemin exact de ton CSV nettoyé."
    )

df = pd.read_csv(INPUT_CSV)

print("Dimensions du dataset :", df.shape)
display(df.head())
print("\\nColonnes disponibles :")
print(df.columns.tolist())

# %%
# ## 2. Vérifier les colonnes nécessaires
#
# On vérifie les noms des colonnes avant de créer les nouvelles variables. Cela évite de continuer silencieusement avec une colonne absente.

# %%
required_candidates = [
    "datetime", "temp", "hum", "moistureLevel",
    "totalLitres", "WateringStatus", "pres", "waterFlow"
]

print("Colonnes présentes :")
for col in required_candidates:
    print(f"- {col}: {'OK' if col in df.columns else 'absente'}")

# Copie de travail : le dataset original chargé reste inchangé
df_fe = df.copy()

# %%
# ## 3. Créer les variables temporelles
#
# La colonne `datetime` est utilisée lorsqu'elle existe. On extrait l'heure, le jour, le mois et le jour de la semaine.
#
# La représentation sin/cos de l'heure tient compte du caractère cyclique des heures (23 h est proche de 0 h).

# %%
if "datetime" not in df_fe.columns:
    raise KeyError("La colonne 'datetime' est absente. Vérifie le nom de la colonne de date.")

df_fe["datetime"] = pd.to_datetime(df_fe["datetime"], errors="coerce")

df_fe["hour"] = df_fe["datetime"].dt.hour
df_fe["day"] = df_fe["datetime"].dt.day
df_fe["month"] = df_fe["datetime"].dt.month
df_fe["day_of_week"] = df_fe["datetime"].dt.dayofweek

# 1 = samedi/dimanche, 0 = lundi-vendredi
df_fe["is_weekend"] = df_fe["day_of_week"].isin([5, 6]).astype("Int64")

# Encodage cyclique de l'heure
df_fe["hour_sin"] = np.sin(2 * np.pi * df_fe["hour"] / 24)
df_fe["hour_cos"] = np.cos(2 * np.pi * df_fe["hour"] / 24)

display(df_fe[[
    "datetime", "hour", "day", "month", "day_of_week",
    "is_weekend", "hour_sin", "hour_cos"
]].head())

# %%
# ## 4. Examiner les cibles avant de préparer les jeux de données
#
# `WateringStatus` contient, d'après l'analyse précédente, les valeurs 0, 1 et 251. On affiche à nouveau la distribution. On ne conserve pour la classification que les classes 0 et 1, à condition que la documentation confirme qu'elles correspondent bien aux deux états d'irrigation.
#
# `target` n'est pas utilisée ici : elle est très majoritairement manquante et sa signification doit être confirmée.

# %%
if "WateringStatus" in df_fe.columns:
    print("Distribution de WateringStatus (avec NaN) :")
    display(df_fe["WateringStatus"].value_counts(dropna=False).sort_index())
else:
    print("Attention : WateringStatus est absente.")

if "totalLitres" in df_fe.columns:
    print("\\nRésumé de totalLitres :")
    display(df_fe["totalLitres"].describe())
else:
    print("Attention : totalLitres est absente.")

# %%
# ## 5. Préparer le dataset de classification
#
# Cible : `WateringStatus`.
#
# Les variables `waterFlow` et `totalLitres` ne sont volontairement pas utilisées comme prédicteurs dans cette première version : elles peuvent être directement liées au fonctionnement de l'irrigation et provoquer une fuite de données (*data leakage*).
#
# Les lignes dont la cible est différente de 0 ou 1 sont exclues de ce jeu de classification provisoire. Il faut confirmer la signification métier de ces codes avant l'entraînement.

# %%
classification_features = [
    "temp",
    "hum",
    "moistureLevel",
    "pres",
    "hour",
    "month",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos"
]

if "WateringStatus" not in df_fe.columns:
    raise KeyError("La colonne cible 'WateringStatus' est absente.")

missing_features = [c for c in classification_features if c not in df_fe.columns]
if missing_features:
    raise KeyError(f"Variables de classification absentes : {missing_features}")

# Ne garder que les lignes avec une classe connue 0 ou 1
mask_binary = df_fe["WateringStatus"].isin([0, 1])

df_classification = df_fe.loc[
    mask_binary, classification_features + ["WateringStatus"]
].copy()

print("Dimensions classification :", df_classification.shape)
print("Distribution de la cible :")
display(df_classification["WateringStatus"].value_counts(dropna=False))

# %%
# ## 6. Préparer le dataset de régression
#
# Cible : `totalLitres`.
#
# Cette première préparation utilise les variables environnementales et temporelles. Elle ne suppose pas que `totalLitres` soit une consommation par heure : il faudra vérifier dans la documentation si cette colonne est instantanée ou cumulative avant de valider la problématique de régression.
#
# Les lignes sans valeur cible sont écartées, mais les valeurs manquantes des variables explicatives sont conservées pour le preprocessing ultérieur.

# %%
regression_features = [
    "temp",
    "hum",
    "moistureLevel",
    "pres",
    "hour",
    "month",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos"
]

if "totalLitres" not in df_fe.columns:
    raise KeyError("La colonne cible 'totalLitres' est absente.")

missing_features = [c for c in regression_features if c not in df_fe.columns]
if missing_features:
    raise KeyError(f"Variables de régression absentes : {missing_features}")

df_regression = df_fe.loc[
    df_fe["totalLitres"].notna(),
    regression_features + ["totalLitres"]
].copy()

print("Dimensions régression :", df_regression.shape)
display(df_regression.head())
print("\\nDistribution de la cible :")
display(df_regression["totalLitres"].describe())

# %%
# ## 7. Préparer le dataset de détection d'anomalies
#
# La détection d'anomalies est non supervisée : aucune colonne cible n'est nécessaire.
#
# On utilise les mesures des capteurs disponibles. Les valeurs manquantes sont conservées ici ; elles devront être imputées dans le pipeline avant d'utiliser un algorithme comme Isolation Forest.

# %%
anomaly_features = [
    "temp",
    "hum",
    "moistureLevel",
    "pres",
    "waterFlow"
]

missing_features = [c for c in anomaly_features if c not in df_fe.columns]
if missing_features:
    raise KeyError(f"Variables de détection d'anomalies absentes : {missing_features}")

df_anomaly = df_fe[anomaly_features].copy()

print("Dimensions anomalies :", df_anomaly.shape)
display(df_anomaly.head())

# %%
# ## 8. Préparer le dataset de clustering
#
# Le clustering est une méthode non supervisée : l'algorithme cherche des groupes d'observations similaires sans utiliser de cible.
#
# On utilise les mesures environnementales `temp`, `hum`, `moistureLevel` et `pres`. On n'inclut pas `WateringStatus`, `target`, `id` ou les colonnes temporelles comme variables de regroupement. Les valeurs manquantes sont conservées ici ; elles seront imputées et les variables standardisées dans le pipeline K-Means/DBSCAN, après cette étape.
#

# %%
clustering_features = [
    "temp",
    "hum",
    "moistureLevel",
    "pres"
]

missing_features = [c for c in clustering_features if c not in df_fe.columns]
if missing_features:
    raise KeyError(f"Variables de clustering absentes : {missing_features}")

df_clustering = df_fe[clustering_features].copy()

print("Dimensions clustering :", df_clustering.shape)
print("\nValeurs manquantes :")
display(df_clustering.isna().sum().to_frame("valeurs_manquantes"))
display(df_clustering.head())

# %%
# **Important :** ne fais pas `fit_transform` de l'imputation ou du scaler sur tout le dataset dans ce notebook. Pour comparer correctement les algorithmes, ces transformations seront intégrées au pipeline de clustering. K-Means et DBSCAN sont sensibles aux échelles des variables.
#

# %%
# ## 9. Contrôler les valeurs manquantes dans les jeux préparés
#
# On mesure les valeurs manquantes sans les remplacer automatiquement. L'imputation (médiane, stratégie temporelle, etc.) sera choisie dans le preprocessing et ajustée uniquement sur les données d'entraînement.

# %%
datasets = {
    "Classification": df_classification,
    "Régression": df_regression,
    "Anomalies": df_anomaly,
    "Clustering": df_clustering
}

for name, data in datasets.items():
    print(f"\n--- {name} ---")
    print("Dimensions :", data.shape)
    display(data.isna().sum().to_frame("valeurs_manquantes"))

# %%
# ## 10. Sauvegarder les datasets issus du Feature Engineering
#
# Les fichiers sont enregistrés dans `../dataset/`. Crée ce dossier s'il n'existe pas.

# %%
OUTPUT_DIR = Path("../dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_fe.to_csv(OUTPUT_DIR / "feature_engineered.csv", index=False)
df_classification.to_csv(OUTPUT_DIR / "classification_data.csv", index=False)
df_regression.to_csv(OUTPUT_DIR / "regression_data.csv", index=False)
df_anomaly.to_csv(OUTPUT_DIR / "anomaly_data.csv", index=False)
df_clustering.to_csv(OUTPUT_DIR / "clustering_data.csv", index=False)

print("Fichiers sauvegardés dans :", OUTPUT_DIR.resolve())
for filename in [
    "feature_engineered.csv",
    "classification_data.csv",
    "regression_data.csv",
    "anomaly_data.csv",
    "clustering_data.csv"
]:
    print("-", filename)

# %%
# ## 11. Points à confirmer avant le Machine Learning
#
# - Vérifier dans la documentation que `WateringStatus` 0 et 1 représentent bien les états attendus ; comprendre le code 251.
# - Comprendre la signification de `totalLitres` : mesure instantanée ou compteur cumulatif.
# - Vérifier les unités et les plages physiques possibles de `temp`, `hum`, `moistureLevel` et `pres`.
# - Vérifier la fréquence des mesures et l'ordre chronologique.
# - Pour les séries temporelles, prévoir une séparation chronologique train/test afin d'éviter d'entraîner le modèle sur des observations futures.
# - Faire l'imputation et la standardisation dans les pipelines des modèles, ajustés uniquement sur les données utilisées pour l'entraînement.
# - Pour le clustering, interpréter les groupes à partir des valeurs originales des capteurs ; les groupes ne sont pas automatiquement des catégories métier validées.
#
# **Fin du Feature Engineering – étape suivante : preprocessing et entraînement des modèles Machine Learning.**
#
