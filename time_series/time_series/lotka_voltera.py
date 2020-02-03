import numpy as np

from time_series.species import SpeciesCollection


class LotkaVolteraModel:
    def __init__(
            self,
            species_collection: SpeciesCollection,
            capacity: float = 1.0
    ):
        """
        A model for the evolution of a population based on interactions in that population.

        Parameters
        ----------
        species_collection
            A population comprising species each of which has a defined growth rate and
            interaction with other species.
        capacity
            The capacity of the environment. A higher capacity allows populations to grow to
            greater numbers.
        """
        self.species_collection = species_collection
        self.capacity = capacity

    def growth_rates(
            self,
            population: np.ndarray
    ) -> np.ndarray:
        """
        Compute the growth rates for a population by taking into account
        interactions and capacity.

        Parameters
        ----------
        population
            A vector of floats describing the current abundance of each species.

        Returns
        -------
        A vector of coefficients describing the growth rate of each species.
        """
        return 1 - self.species_collection.interaction_matrix.dot(
            population
        ) / self.capacity

    def change(
            self,
            population: np.ndarray
    ) -> np.ndarray:
        """
        Compute the change in population for each species.

        Parameters
        ----------
        population
            A vector of floats describing the current abundance of each species.

        Returns
        -------
        A vector of values describing the change in abundance of each species.
        """
        population_coefficients = population * self.species_collection.growth_rate_vector
        return population_coefficients * self.growth_rates(population)

    def step(
            self,
            population: np.ndarray
    ) -> np.ndarray:
        """
        Compute a new population for each species (t = t + 1)

        Parameters
        ----------
        population
            A vector of floats describing the current abundance of each species.

        Returns
        -------
        A vector of floats describing the new abundance of each species.
        """
        return np.array(population) + self.change(population)
