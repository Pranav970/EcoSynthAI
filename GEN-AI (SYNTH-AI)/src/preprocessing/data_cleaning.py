import pandas as pd
import numpy as np

def clean_biodiversity_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate entries
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    df.fillna({
        'common_name': 'Unknown',
        'habitat': 'Unspecified',
        'conservation_status': 'Not Evaluated'
    }, inplace=True)
    
    # Standardize text columns
    text_columns = ['scientific_name', 'common_name', 'habitat']
    for col in text_columns:
        df[col] = df[col].str.strip().str.lower()
    
    return df

def normalize_climate_data(df: pd.DataFrame) -> pd.DataFrame:
    # Min-Max scaling for numerical columns
    numerical_cols = ['temperature', 'precipitation', 'extreme_events']
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())
    
    return df