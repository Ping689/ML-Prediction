import pandas as pd
import numpy as np
import bentoml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    df = pd.read_csv('prepared_dataset.csv')

    TARGET = 'SiteEnergyUse(kBtu)'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    y_log = np.log1p(y)

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    final_rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=30,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', final_rf_model)
    ])

    final_pipeline.fit(X, y_log)

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