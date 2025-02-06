import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class SoilDataProcessor:
    def __init__(self, data_dir: str = 'data/raw/soil_data/'):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()

    def load_soil_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        all_data = []
        try:
            for file_path in self.data_dir.glob(file_pattern):
                df = pd.read_csv(file_path)
                all_data.append(df)
            return pd.concat(all_data, ignore_index=True)
        except Exception as e:
            self.logger.error(f"Error loading soil data: {str(e)}")
            return pd.DataFrame()

    def process_soil_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Handle missing values
            df['ph'] = df['ph'].fillna(df['ph'].mean())
            df['organic_matter'] = df['organic_matter'].fillna(df['organic_matter'].median())
            
            # Normalize numerical columns
            numerical_cols = ['ph', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter']
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            
            return df
        except Exception as e:
            self.logger.error(f"Error processing soil data: {str(e)}")
            return df