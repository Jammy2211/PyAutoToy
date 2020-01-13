from typing import List

import numpy as np

from time_series import matrix as m


class Species(m.Species):
    def __init__(
            self,
            interactions=None,
            growth_rate: float = 1.0,
            observables=None
    ):
        """
        A species that has a defined growth rate and interaction rate with other species.

        If no interaction rate is defined then a default interaction rate of 0.0 is used.

        Parameters
        ----------
        growth_rate
            The rate of growth of the species in the absence of other species.
        """
        super().__init__(
            interactions
        )
        self.growth_rate = growth_rate
        self.observables = observables or dict()


class SpeciesCollection(m.Matrix):
    @property
    def items(self):
        return self.species

    def __init__(self, species: List[Species]):
        """
        A collection of species which interact with each other.

        If no interaction is defined between two species it defaults to 0.

        Parameters
        ----------
        species
            A list of species.
        """
        self.species = list(species)
        for species_a in self.species:
            for species_b in self.species:
                if species_b not in species_a.interactions:
                    species_a.interactions[species_b] = 0.0

    @property
    def growth_rate_vector(self) -> np.ndarray:
        """
        A vector of floats describing the growth rate of each individual species.
        """
        return np.array([
            species.growth_rate
            for species in self.species
        ])
