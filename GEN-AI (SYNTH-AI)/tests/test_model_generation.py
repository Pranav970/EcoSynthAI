import unittest
import torch
import numpy as np
from models.gan.species_generator import SpeciesGenerator
from models.gan.ecosystem_discriminator import EcosystemDiscriminator
from models.transformers.trait_encoder import TraitEncoder

class TestSpeciesGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SpeciesGenerator()
        
    def test_species_generation(self):
        noise = torch.randn(1, 100)
        conditions = {
            'temperature': 25.0,
            'rainfall': 1000,
            'soil_ph': 6.5
        }
        generated_species = self.generator.generate(noise, conditions)
        self.assertIsInstance(generated_species, dict)
        self.assertIn('dna_sequence', generated_species)
        self.assertIn('traits', generated_species)
        
    def test_environmental_constraints(self):
        extreme_conditions = {
            'temperature': 50.0,  # Very hot
            'rainfall': 0,        # Very dry
            'soil_ph': 2.0       # Very acidic
        }
        noise = torch.randn(1, 100)
        species = self.generator.generate(noise, extreme_conditions)
        self.assertTrue(self.generator.validate_survival_probability(species, extreme_conditions))

class TestEcosystemDiscriminator(unittest.TestCase):
    def setUp(self):
        self.discriminator = EcosystemDiscriminator()
        
    def test_ecosystem_stability(self):
        test_ecosystem = {
            'species': [
                {'type': 'producer', 'traits': {...}},
                {'type': 'consumer', 'traits': {...}},
                {'type': 'decomposer', 'traits': {...}}
            ],
            'interactions': [...]
        }
        stability_score = self.discriminator.evaluate_stability(test_ecosystem)
        self.assertGreaterEqual(stability_score, 0.0)
        self.assertLessEqual(stability_score, 1.0)
