from collections import defaultdict
from typing import Union, Tuple

import numpy as np


class Species:
    def __init__(self, growth_rate: float):
        """
        A species that has a defined growth rate and interaction rate with other species.

        If no interaction rate is defined then a default interaction rate of 1.0 is used.

        Parameters
        ----------
        growth_rate
            The rate of growth of the species in the absence of other species.
        """
        self.interactions = defaultdict(lambda: 1.0)
        self.growth_rate = growth_rate

    def __getitem__(self, species: "Species") -> float:
        """
        Convenience method for getting the interaction value of this species with another.

        If no interaction rate has been set then 1.0 is returned.

        Parameters
        ----------
        species
            A species with which this species interacts

        Returns
        -------
        The negative impact the presence of the other species has on the growth
        of this species.
        """
        return self.interactions[species]

    def __setitem__(
            self,
            species: "Species",
            interaction: float
    ):
        """
        Convenience method for setting the interaction value of this species with another.

        Parameters
        ----------
        species
            A species with which this species interacts
        interaction
            The negative impact the presence of the other species has on the growth
            of this species.
        """
        self.interactions[species] = interaction


Index = Union[int, Tuple[int, int]]
SpeciesOrInteraction = Union[Species, float]


class SpeciesCollection:
    def __init__(self, *species):
        """
        A collection of species which interact with each other.

        Parameters
        ----------
        species
            A list of species.
        """
        self.species = list(species)
        for species_a in self.species:
            for species_b in self.species:
                if species_b not in species_a.interactions:
                    species_a.interactions[species_b] = 1.0

    @property
    def interaction_matrix(self) -> np.ndarray:
        """
        A 2D matrix of floats describing the interactions between species in the collection.

        The diagonal is the self-interaction of the species.
        """
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
    def growth_rate_vector(self) -> np.ndarray:
        """
        A vector of floats describing the growth rate of each individual species.
        """
        return np.array([
            species.growth_rate
            for species in self.species
        ])

    def __getitem__(
            self,
            item: Index
    ) -> SpeciesOrInteraction:
        """
        If an integer is passed in then a species is returned.

        If a tuple of integers is passed in then the interaction between the ith and jth species
        is returned.

        Parameters
        ----------
        item
            Index arguments (int or tuple)

        Returns
        -------
        A species or interaction coefficient.
        """
        if isinstance(item, int):
            return self.species[item]
        elif isinstance(item, tuple):
            return self[item[0]][self[item[1]]]

    def __setitem__(
            self,
            index: Index,
            value: SpeciesOrInteraction
    ):
        """
        If the index is an integer then a species is places in that index of the species list.

        If the index is a tuple then the interaction i, j is set to value.

        Parameters
        ----------
        index
            A tuple or integer addressing an interaction or species
        value
            A float or species
        """
        if isinstance(index, int):
            self.species[index] = value
        elif isinstance(index, tuple):
            self[index[0]][self[index[1]]] = value
