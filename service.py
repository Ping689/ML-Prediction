
import numpy as np
import pandas as pd
import bentoml
from pydantic import BaseModel, Field

class BuildingData(BaseModel):
    PrimaryPropertyType: str = Field(..., example="Office")
    NumberofFloors: int = Field(..., gt=0, example=12)
    PropertyGFATotal: float = Field(..., gt=0, example=120000.0)
    BuildingAge: int = Field(..., gt=0, example=15)
    GFAPerFloor: float = Field(..., gt=0, example=10000.0)
    parking_ratio: float = Field(..., ge=0, le=1, example=0.1)
    NumberOfUseTypes: int = Field(..., gt=0, example=2)
    DistanceFromCenter: float = Field(..., ge=0, example=5.5)

    class Config:
        arbitrary_types_allowed = True

# On charge la référence du modèle RandomForest
model_ref = bentoml.sklearn.get("seattle_energy_forest_regressor:latest")

# On charge le véritable pipeline scikit-learn depuis la référence
final_model = model_ref.load_model()

@bentoml.service
class SeattleEnergyService:

    @bentoml.api
    def predict(self, building_data: BuildingData) -> dict:
        """
        Endpoint de prédiction pour le modèle RandomForest final.
        """
        input_dict = building_data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        model_columns = [
            "PrimaryPropertyType", "NumberofFloors", "PropertyGFATotal",
            "BuildingAge", "GFAPerFloor", "parking_ratio",
            "NumberOfUseTypes", "DistanceFromCenter",
        ]
        input_df = input_df[model_columns]

        log_prediction = final_model.predict(input_df)
        
        prediction = np.expm1(log_prediction[0])
        
        return {
            "predicted_site_energy_use_kbtu": round(prediction, 2)
        }
