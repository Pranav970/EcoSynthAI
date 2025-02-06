import scrapy
import pandas as pd
import json
from typing import List, Dict

class BiodiversityScraper(scrapy.Spider):
    name = 'biodiversity_crawler'
    
    def __init__(self, regions: List[str], output_file: str):
        self.regions = regions
        self.output_file = output_file
        self.collected_data = []
    
    def start_requests(self):
        for region in self.regions:
            yield scrapy.Request(
                url=f'https://gbif.org/species/region/{region}',
                callback=self.parse_biodiversity
            )
    
    def parse_biodiversity(self, response):
        species_data = response.css('.species-entry')
        
        for species in species_data:
            species_info = {
                'scientific_name': species.css('.scientific-name::text').get(),
                'common_name': species.css('.common-name::text').get(),
                'habitat': species.css('.habitat::text').get(),
                'conservation_status': species.css('.status::text').get()
            }
            self.collected_data.append(species_info)
    
    def closed(self, reason):
        df = pd.DataFrame(self.collected_data)
        df.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")