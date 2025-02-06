import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.impute import KNNImputer
from datetime import datetime

class DataCleaner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.imputer = KNNImputer(n_neighbors=5)

    def clean_biodiversity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            categorical_cols = ['species_name', 'habitat', 'conservation_status']
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            
            # Categorical imputation
            for col in categorical_cols:
                df[col] = df[col].fillna('Unknown')
            
            # Numerical imputation using KNN
            df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
            
            return df
        except Exception as e:
            self.logger.error(f"Error cleaning biodiversity data: {str(e)}")
            return df

    def clean_climate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Remove outliers using IQR method
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            
            return df
        except Exception as e:
            self.logger.error(f"Error cleaning climate data: {str(e)}")
            return df