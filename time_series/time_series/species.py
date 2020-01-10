from collections import defaultdict

import numpy as np


class Species:
    def __init__(self, growth_rate):
        self.interactions = defaultdict(lambda: 1.0)
        self.growth_rate = growth_rate


class SpeciesCollection:
    def __init__(self, *species):
        self.species = species
        for species_a in self.species:
            for species_b in self.species:
                if species_b not in species_a.interactions:
                    species_a.interactions[species_b] = 1.0

    @property
    def interaction_matrix(self):
        return np.array([
            [
                species_a.interactions[
                    species_b
                ]
                for species_b in self.species
            ]
            for species_a in self.species
        ])

    @property
    def growth_rate_vector(self):
        return np.array([
            species.growth_rate
            for species in self.species
        ])
