from typing import List, Set

import numpy as np

from time_series import matrix as m
from time_series.observable import CompoundObservable
from time_series.util import assert_lengths_match


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
        observables
            A dictionary relating the names of observables to their distributions with
            respect to this species.
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


class SpeciesObservables:
    @assert_lengths_match
    def __init__(
            self,
            abundances: List[float],
            species: List[Species]
    ):
        """
        Relates a list of relative abundances to associated species.

        Parameters
        ----------
        abundances
            A list of floats indicating the abundance of each species
        species
            A list of species
        """
        self.abundances = abundances
        self.species = species

    @property
    def observable_names(self) -> Set[str]:
        """
        The names of all the observables found in the list of species
        """
        return {
            key for species
            in self.species
            for key in species.observables.keys()
        }

    def __getitem__(self, name: str) -> CompoundObservable:
        """
        Get a CompoundObservable which comprises an observable of the same name from each species.

        Assumes every species has an observable with the given name.

        Parameters
        ----------
        name
            The name of the observable

        Returns
        -------
        A compound observable comprising all observables with the given name
        """
        observables = [
            species.observables[name]
            for species in self.species
        ]
        return CompoundObservable(
            self.abundances,
            observables
        )
