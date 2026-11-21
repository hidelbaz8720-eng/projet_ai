# # Data Cleaning — version corrigée
#
# ## Objectif
# Préparer le dataset des capteurs pour l'analyse et le Machine Learning, en conservant les informations utiles et en évitant de supprimer automatiquement des observations potentiellement valides.
#
# **Ordre de travail :** chargement → inspection → doublons → valeurs manquantes → types → contrôle des valeurs aberrantes → validation métier → export.

# %%
# ## 1. Importation des bibliothèques et chargement
#
# Modifie `CSV_PATH` pour indiquer l'emplacement réel de ton fichier CSV. Le fichier doit être accessible depuis le notebook.

# %%
import pandas as pd
import numpy as np
from pathlib import Path

# À MODIFIER : chemin du fichier CSV sur ton ordinateur
CSV_PATH = Path(r"C:/Users/pc/Documents/IA cours/projet/export_2023-12-31-23-00_to_2026-11-14-22-59.csv")

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {CSV_PATH}\n"
        "Corrige CSV_PATH avec le chemin réel de ton fichier CSV."
    )

df = pd.read_csv(CSV_PATH)

# Conserver les données brutes sans les modifier
df_original = df.copy(deep=True)
df_clean = df.copy(deep=True)

print("Dataset original :", df_original.shape)
print("Dataset à nettoyer :", df_clean.shape)
display(df_clean.head())

# %%
# ## 2. Inspection initiale
# Cette étape permet de connaître les dimensions, les noms de colonnes, les types et les statistiques descriptives avant toute modification.

# %%
print("Dimensions :", df_clean.shape)
print("\nColonnes :")
print(df_clean.columns.tolist())

print("\nTypes et valeurs non nulles :")
df_clean.info()

display(df_clean.describe(include="all").T)

# %%
# ## 3. Vérification des doublons
# On compte les lignes dupliquées puis on les supprime. On conserve la première occurrence.

# %%
nb_doublons = df_clean.duplicated().sum()
print("Nombre de doublons avant suppression :", nb_doublons)

df_clean = df_clean.drop_duplicates().copy()

print("Dimensions après suppression :", df_clean.shape)
print("Doublons restants :", df_clean.duplicated().sum())

# %%
# ## 4. Analyse des valeurs manquantes
#
# On calcule le nombre et le pourcentage de valeurs manquantes par colonne.
#
# **Important :** on ne supprime pas immédiatement une colonne parce qu'elle contient beaucoup de valeurs manquantes. La décision dépend de son rôle dans le projet. En particulier, `target` doit être conservée dans le dataset de travail si elle représente la variable à prédire.

# %%
missing = pd.DataFrame({
    "Valeurs manquantes": df_clean.isna().sum(),
    "Pourcentage (%)": (df_clean.isna().mean() * 100).round(2)
}).sort_values("Pourcentage (%)", ascending=False)

display(missing)

# %%
# ### 4.1 Cas de la variable cible `target`
#
# On garde `target` dans `df_clean`. Pour l'apprentissage supervisé, on crée séparément `df_supervised`, qui contient uniquement les lignes où `target` est renseignée.
#
# Ne fais pas `df_clean = df_clean.drop(columns=["target"])` avant cette étape.

# %%
if "target" in df_clean.columns:
    print("Valeurs disponibles dans target :")
    display(df_clean["target"].value_counts(dropna=False))

    print("Valeurs manquantes dans target :", df_clean["target"].isna().sum())

    df_supervised = df_clean.dropna(subset=["target"]).copy()
    print("Dataset supervisé :", df_supervised.shape)
else:
    print("Attention : la colonne 'target' est absente. Vérifie le nom exact de la variable cible.")
    df_supervised = None

# %%
# ## 5. Vérification et conversion des types
#
# Les variables de mesure doivent être numériques pour les statistiques et les modèles. Les valeurs non convertibles deviennent `NaN` et seront comptabilisées dans le bilan.

# %%
variables = [
    "temp", "hum", "moistureLevel",
    "totalLitres", "pres", "waterFlow"
]

colonnes_numeriques_presentes = [c for c in variables if c in df_clean.columns]
colonnes_absentes = [c for c in variables if c not in df_clean.columns]

if colonnes_absentes:
    print("Colonnes de mesure absentes :", colonnes_absentes)

for col in colonnes_numeriques_presentes:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

print("Types après conversion :")
display(df_clean[colonnes_numeriques_presentes].dtypes)

# %%
# ## 6. Détection des valeurs aberrantes avec la méthode IQR
#
# Pour chaque variable numérique :
# - Q1 est le 1er quartile (25 %).
# - Q3 est le 3e quartile (75 %).
# - IQR = Q3 − Q1.
# - Limite basse = Q1 − 1,5 × IQR.
# - Limite haute = Q3 + 1,5 × IQR.
#
# Les valeurs hors de ces limites sont signalées comme **outliers potentiels**. Cette méthode sert ici au diagnostic : elle ne supprime ni ne remplace les valeurs.

# %%
resultats_iqr = []

for col in colonnes_numeriques_presentes:
    serie = df_clean[col].dropna()

    if serie.empty:
        resultats_iqr.append({
            "Variable": col, "Q1": np.nan, "Q3": np.nan,
            "IQR": np.nan, "Limite basse": np.nan,
            "Limite haute": np.nan, "Nombre outliers": 0,
            "Pourcentage outliers (%)": 0.0
        })
        continue

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    masque_outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
    nb_outliers = int(masque_outliers.sum())
    nb_valides = int(df_clean[col].notna().sum())
    pct = (nb_outliers / nb_valides * 100) if nb_valides else 0.0

    resultats_iqr.append({
        "Variable": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Limite basse": lower,
        "Limite haute": upper,
        "Nombre outliers": nb_outliers,
        "Pourcentage outliers (%)": round(pct, 2)
    })

df_iqr = pd.DataFrame(resultats_iqr)
display(df_iqr)

# %%
# ### 6.1 Visualisation des distributions et des outliers
#
# Les boxplots permettent de visualiser les valeurs centrales, la dispersion et les points éloignés. Ils ne prouvent pas à eux seuls qu'une mesure est erronée.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

if colonnes_numeriques_presentes:
    n = len(colonnes_numeriques_presentes)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.8 * n)))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, colonnes_numeriques_presentes):
        sns.boxplot(x=df_clean[col], ax=ax)
        ax.set_title(f"Boxplot — {col}")
        ax.set_xlabel(col)

    plt.tight_layout()
    plt.show()

# %%
# ## 7. Contrôles de cohérence métier (à confirmer)
#
# Les bornes IQR sont statistiques et dépendent de la distribution. Elles ne doivent pas être confondues avec les limites physiques d'un capteur.
#
# Les règles ci-dessous sont donc **optionnelles**. Active-les uniquement après confirmation des unités et des plages valides dans la documentation des capteurs. Une valeur invalide est remplacée par `NaN`, pas par une valeur inventée.

# %%
# Exemple de règles à valider avec la documentation technique.
# Laisser False tant que les unités et limites des capteurs ne sont pas confirmées.
APPLIQUER_REGLES_METIER = False

if APPLIQUER_REGLES_METIER:
    regles = {
        "temp": (-50, 100),
        "hum": (0, 100),
        "moistureLevel": (0, 1024),
        # Ajouter totalLitres, pres et waterFlow seulement si leurs bornes
        # physiques sont connues et confirmées.
    }

    for col, (minimum, maximum) in regles.items():
        if col in df_clean.columns:
            masque_invalide = (df_clean[col] < minimum) | (df_clean[col] > maximum)
            print(f"{col} : {int(masque_invalide.sum())} mesure(s) hors limites")
            df_clean.loc[masque_invalide, col] = np.nan
else:
    print("Règles métier non appliquées : bornes à confirmer.")

# %%
# ## 8. Vérification des variables catégorielles
#
# On examine les modalités de `WateringStatus` et `messageType` sans supprimer de code automatiquement. Une valeur inhabituelle (par exemple 251) doit être interprétée à partir de la documentation du système.

# %%
for col in ["WateringStatus", "messageType"]:
    if col in df_clean.columns:
        print(f"\nValeurs de {col} :")
        display(df_clean[col].value_counts(dropna=False).head(30))

# %%
# ## 9. Conversion de la date
#
# On convertit `datetime` en type date/heure. `errors="coerce"` transforme les dates illisibles en `NaT`. `utc=True` interprète/convertit les dates en UTC ; vérifie que cela correspond au fuseau horaire de la source.

# %%
if "datetime" in df_clean.columns:
    df_clean["datetime"] = pd.to_datetime(
        df_clean["datetime"],
        errors="coerce",
        utc=True
    )

    print("Type :", df_clean["datetime"].dtype)
    print("Dates invalides :", df_clean["datetime"].isna().sum())
    print("Première date :", df_clean["datetime"].min())
    print("Dernière date :", df_clean["datetime"].max())
else:
    print("La colonne datetime est absente.")

# %%
# ## 10. Bilan final du nettoyage
#
# On vérifie les dimensions, les doublons, les valeurs manquantes et les types après nettoyage.

# %%
print("Dimensions finales :", df_clean.shape)
print("Doublons restants :", df_clean.duplicated().sum())

bilan_final = pd.DataFrame({
    "Valeurs manquantes": df_clean.isna().sum(),
    "Pourcentage (%)": (df_clean.isna().mean() * 100).round(2),
    "Type": df_clean.dtypes.astype(str)
})
display(bilan_final)

# %%
# ## 11. Export des datasets
#
# - `dataset_cleaned.csv` : dataset nettoyé complet, avec `target` conservée.
# - `dataset_supervised.csv` : lignes dont `target` est renseignée, si la colonne existe.
#
# Les deux fichiers sont enregistrés dans le dossier courant du notebook.

# %%
df_clean.to_csv("dataset_cleaned.csv", index=False)
print("Dataset nettoyé sauvegardé : dataset_cleaned.csv")

if df_supervised is not None and "target" in df_clean.columns:
    # Recréer le dataset supervisé à partir du dataset final nettoyé
    df_supervised = df_clean.dropna(subset=["target"]).copy()
    df_supervised.to_csv("dataset_supervised.csv", index=False)
    print("Dataset supervisé sauvegardé : dataset_supervised.csv")

# %%
# ## Conclusion méthodologique
#
# Le nettoyage comprend la suppression des doublons, l'analyse des valeurs manquantes, la conversion des types, la détection statistique des outliers et la conversion des dates. La méthode IQR est utilisée pour **repérer** les valeurs extrêmes, et non pour les supprimer automatiquement. Les règles physiques doivent être confirmées par la documentation des capteurs avant d'être appliquées.
#
# La variable `target` est conservée dans le dataset complet. Les lignes sans cible sont écartées uniquement de la copie destinée à l'apprentissage supervisé.
