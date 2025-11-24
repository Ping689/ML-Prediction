import pandas as pd
import numpy as np
import bentoml

# Import des modules scikit-learn nécessaires
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    # Charger les données préparées depuis le notebook
    df = pd.read_csv('prepared_dataset.csv')

    # Définir la cible (y) et les features (X)
    TARGET = 'SiteEnergyUse(kBtu)'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Appliquer la même transformation log que dans le notebook
    y_log = np.log1p(y)

    # Recréer le pipeline de pré-traitement 
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Définir le modèle final avec les meilleurs hyperparamètres
    final_rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    # Créer le pipeline complet : pré-processeur + modèle
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', final_rf_model)
    ])

    # Entraîner le pipeline final sur l'ensemble des données
    final_pipeline.fit(X, y_log)

    # Sauvegarder le pipeline entraîné avec BentoML
    bento_model = bentoml.sklearn.save_model(
        "seattle_energy_forest_regressor", 
        final_pipeline,
        metadata={
            "description": "RandomForestRegressor optimisé pour la consommation d'énergie de Seattle.",
            "features": X.columns.tolist()
        }
    )
except FileNotFoundError:
    print("ERREUR: Le fichier 'prepared_dataset.csv' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")