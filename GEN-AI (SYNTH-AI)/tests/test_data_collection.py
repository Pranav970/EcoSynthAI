import unittest
import pandas as pd
from unittest.mock import Mock, patch
from src.data_collection.biodiversity_scraper import BiodiversityScraper
from src.data_collection.climate_api_handler import ClimateAPIHandler
from src.data_collection.soil_data_processor import SoilDataProcessor

class TestBiodiversityScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = BiodiversityScraper()
        
    def test_species_data_format(self):
        sample_data = self.scraper.get_species_data("test_region")
        self.assertIsInstance(sample_data, pd.DataFrame)
        required_columns = ['species_name', 'family', 'traits', 'native_range']
        for col in required_columns:
            self.assertIn(col, sample_data.columns)
            
    @patch('src.data_collection.biodiversity_scraper.requests.get')
    def test_api_error_handling(self, mock_get):
        mock_get.side_effect = Exception("API Error")
        with self.assertRaises(Exception):
            self.scraper.get_species_data("test_region")

class TestClimateAPIHandler(unittest.TestCase):
    def setUp(self):
        self.api_handler = ClimateAPIHandler()
        
    def test_climate_data_validation(self):
        climate_data = self.api_handler.get_climate_data(
            latitude=40.7128,
            longitude=-74.0060,
            start_date="2020-01-01",
            end_date="2020-12-31"
        )
        self.assertIsInstance(climate_data, dict)
        self.assertIn('temperature', climate_data)
        self.assertIn('precipitation', climate_data)
        self.assertIn('humidity', climate_data)

class TestSoilDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = SoilDataProcessor()
        
    def test_soil_composition_analysis(self):
        test_data = {
            'ph': 7.0,
            'organic_matter': 5.0,
            'nitrogen': 1.5,
            'phosphorus': 0.5,
            'potassium': 2.0
        }
        result = self.processor.analyze_soil_composition(test_data)
        self.assertIn('soil_quality_score', result)
        self.assertIn('nutrient_deficiencies', result)
        self.assertIsInstance(result['nutrient_deficiencies'], list)
