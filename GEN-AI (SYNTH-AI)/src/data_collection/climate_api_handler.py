import requests
import pandas as pd
from typing import List, Dict

class ClimateDataCollector:
    def __init__(self, api_key: str):
        self.base_url = "https://climate-api.example.com/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def fetch_climate_data(self, regions: List[str], years: List[int]) -> pd.DataFrame:
        all_climate_data = []
        
        for region in regions:
            for year in years:
                response = requests.get(
                    f"{self.base_url}/climate-data",
                    params={"region": region, "year": year},
                    headers=self.headers
                )
                data = response.json()
                all_climate_data.append({
                    'region': region,
                    'year': year,
                    'temperature': data.get('avg_temperature'),
                    'precipitation': data.get('total_precipitation'),
                    'extreme_events': data.get('extreme_events_count')
                })
        
        return pd.DataFrame(all_climate_data)