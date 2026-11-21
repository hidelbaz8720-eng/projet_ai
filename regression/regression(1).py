"""
regression.py
---------------
Module de regression pour predire totalLitres a partir de dataset_cleaned.csv.

Installation :
    pip install pandas numpy scikit-learn matplotlib joblib

Execution :
    python regression.py
    python regression.py --data "C:/chemin/vers/dataset_cleaned.csv"

Le script :
- charge le dataset nettoye ;
- prepare les variables numeriques ;
- entraine plusieurs modeles de regression ;
- compare les modeles avec MAE, MSE, RMSE et R2 ;
- sauvegarde le meilleur modele et les graphiques.

Adapte FEATURES si tu souhaites imposer une liste precise de variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# --------------------------- Configuration ---------------------------

TARGET = "totalLitres"

# Laisser None pour utiliser automatiquement toutes les colonnes numeriques
# pertinentes, sauf la cible. Pour choisir les variables manuellement :
# FEATURES = ["temp", "hum", "moistureLevel", "pres", "waterFlow"]
FEATURES = None

DEFAULT_DATA = Path("dataset_cleaned.csv")
OUTPUT_DIR = Path("regression_results")
MODEL_PATH = OUTPUT_DIR / "best_regression_model.joblib"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV et verifie la presence de la variable cible."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {csv_path.resolve()}\n"
            "Place dataset_cleaned.csv dans le meme dossier que regression.py "
            "ou utilise --data pour indiquer son chemin."
        )

    # sep=None permet de detecter automatiquement , ; ou tabulation.
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = df.columns.astype(str).str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            f"La colonne cible '{TARGET}' est absente.\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

    return df


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Construit X et y en evitant d'utiliser la cible comme entree."""
    data = df.copy()

    # Convertir la cible en numerique et retirer les lignes sans cible.
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET])

    if FEATURES is not None:
        missing = [col for col in FEATURES if col not in data.columns]
        if missing:
            raise ValueError(f"Variables explicatives absentes : {missing}")
        feature_names = list(FEATURES)
    else:
        # Les colonnes texte (par exemple dates au format texte) sont ignorees.
        # Les identifiants sont exclus pour eviter un apprentissage artificiel.
        excluded_names = {
            TARGET.lower(), "id", "index", "unnamed: 0",
            "timestamp", "date", "datetime"
        }
        feature_names = [
            col for col in data.select_dtypes(include=[np.number]).columns
            if col.lower() not in excluded_names
        ]

    if not feature_names:
        raise ValueError(
            "Aucune variable explicative numerique trouvee. "
            "Verifie les colonnes du dataset ou renseigne FEATURES."
        )

    X = data[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data[TARGET].astype(float)

    # Supprimer les variables completement vides.
    feature_names = [col for col in feature_names if not X[col].isna().all()]
    X = X[feature_names]

    if X.shape[1] == 0:
        raise ValueError("Toutes les variables explicatives sont vides.")

    if len(X) < 5:
        raise ValueError(
            f"Seulement {len(X)} lignes exploitables. "
            "Il faut au minimum 5 lignes, et davantage pour une evaluation fiable."
        )

    return X, y, feature_names


def build_models(numeric_features: list[str]) -> Dict[str, Pipeline]:
    """Cree les pipelines de preprocessing et les modeles."""
    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features)
        ],
        remainder="drop",
    )

    return {
        "Linear Regression": Pipeline(
            [
                ("preprocessing", linear_preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessing", tree_preprocessor),
                ("model", DecisionTreeRegressor(
                    max_depth=6, min_samples_leaf=2, random_state=42
                )),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessing", tree_preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessing", tree_preprocessor),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }


def evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Entraine les modeles et calcule les metriques sur le jeu de test."""
    rows = []
    predictions = {}

    for name, pipeline in models.items():
        print(f"\nEntrainement : {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_test, y_pred),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by="R2", ascending=False
    ).reset_index(drop=True)

    return results, predictions


def save_plots(
    results: pd.DataFrame,
    y_test: pd.Series,
    predictions: Dict[str, np.ndarray],
    best_name: str,
    output_dir: Path,
) -> None:
    """Sauvegarde les graphiques de comparaison et de prediction."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparaison des scores R2.
    plt.figure(figsize=(9, 5))
    plt.bar(results["Model"], results["R2"])
    plt.title("Comparaison des modeles - R2")
    plt.ylabel("R2")
    plt.xlabel("Modele")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "comparaison_modeles_r2.png", dpi=150)
    plt.close()

    # Valeurs reelles vs predites pour le meilleur modele.
    best_pred = predictions[best_name]
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_pred, alpha=0.7)
    low = min(float(y_test.min()), float(np.min(best_pred)))
    high = max(float(y_test.max()), float(np.max(best_pred)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title(f"Valeurs reelles vs predites - {best_name}")
    plt.xlabel("Valeurs reelles (totalLitres)")
    plt.ylabel("Valeurs predites (totalLitres)")
    plt.tight_layout()
    plt.savefig(output_dir / "reel_vs_predit.png", dpi=150)
    plt.close()

    # Distribution des residus du meilleur modele.
    residuals = np.asarray(y_test) - best_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.title(f"Distribution des erreurs - {best_name}")
    plt.xlabel("Erreur (valeur reelle - valeur predite)")
    plt.ylabel("Frequence")
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_erreurs.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction de totalLitres avec plusieurs modeles de regression."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Chemin vers le fichier CSV nettoye (defaut : dataset_cleaned.csv).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion des donnees reservees au test (defaut : 0.20).",
    )
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.5:
        raise ValueError("--test-size doit etre compris entre 0.1 et 0.5.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PARTIE MACHINE LEARNING : REGRESSION")
    print(f"Variable a predire : {TARGET}")
    print("=" * 65)

    df = load_dataset(args.data)
    X, y, feature_names = prepare_data(df)

    print(f"\nNombre de lignes utilisees : {len(X)}")
    print(f"Variables explicatives : {feature_names}")
    print(f"Variable cible : {TARGET}")

    # Un jeu de test d'au moins 1 observation.
    test_count = max(1, int(np.ceil(len(X) * args.test_size)))
    if len(X) - test_count < 2:
        raise ValueError("Pas assez de donnees pour separer train et test.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
    )

    print(f"\nDonnees d'entrainement : {len(X_train)}")
    print(f"Donnees de test : {len(X_test)}")

    models = build_models(feature_names)
    results, predictions = evaluate_models(
        models, X_train, X_test, y_train, y_test
    )

    print("\n" + "=" * 65)
    print("RESULTATS DE COMPARAISON")
    print("=" * 65)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Le meilleur modele est celui avec le R2 le plus eleve.
    best_name = str(results.iloc[0]["Model"])
    best_model = models[best_name]

    joblib.dump(
        {
            "model": best_model,
            "target": TARGET,
            "features": feature_names,
            "model_name": best_name,
        },
        MODEL_PATH,
    )

    results.to_csv(OUTPUT_DIR / "resultats_regression.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "valeur_reelle": y_test.to_numpy(),
            "valeur_predite": predictions[best_name],
        },
        index=y_test.index,
    ).sort_index()
    predictions_df.to_csv(
        OUTPUT_DIR / "predictions_meilleur_modele.csv", index_label="index"
    )

    save_plots(results, y_test, predictions, best_name, OUTPUT_DIR)

    summary = {
        "target": TARGET,
        "best_model": best_name,
        "features": feature_names,
        "metrics": results.iloc[0].to_dict(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(OUTPUT_DIR / "resume_regression.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"Meilleur modele : {best_name}")
    print(f"Modele sauvegarde : {MODEL_PATH}")
    print(f"Resultats et graphiques : {OUTPUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
