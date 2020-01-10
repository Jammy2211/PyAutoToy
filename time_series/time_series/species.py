from collections import defaultdict

import numpy as np


class Species:
    def __init__(self, growth_rate):
        self.interactions = defaultdict(lambda: 1.0)
        self.growth_rate = growth_rate

    def __getitem__(self, item):
        return self.interactions[item]

    def __setitem__(self, key, value):
        self.interactions[key] = value


class SpeciesCollection:
    def __init__(self, *species):
        self.species = list(species)
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

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.species[item]
        elif isinstance(item, tuple):
            return self[item[0]][self[item[1]]]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.species[key] = value
        elif isinstance(key, tuple):
            self[key[0]][self[key[1]]] = value
