import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Create interaction features
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['soil_quality_score'] = df.apply(self._calculate_soil_score, axis=1)
            
            # Encode categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            encoded_features = self.encoder.fit_transform(df[categorical_cols])
            
            # Scale numerical features
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            scaled_features = self.scaler.fit_transform(df[numerical_cols])
            
            # Combine features
            final_df = pd.concat([
                pd.DataFrame(scaled_features, columns=numerical_cols),
                pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(categorical_cols))
            ], axis=1)
            
            return final_df
        except Exception as e:
            self.logger.error(f"Error engineering features: {str(e)}")
            return df

    def _calculate_soil_score(self, row: pd.Series) -> float:
        try:
            return (row['ph'] * 0.3 + 
                   row['organic_matter'] * 0.3 + 
                   row['nitrogen'] * 0.2 + 
                   row['phosphorus'] * 0.1 + 
                   row['potassium'] * 0.1)
        except KeyError:
            return np.nan