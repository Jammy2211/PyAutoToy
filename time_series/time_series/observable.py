from typing import List

import numpy as np
from scipy import stats

from time_series.util import assert_lengths_match
from abc import ABC, abstractmethod


class AbstractObservable(ABC):
    @abstractmethod
    def pdf(self, lower_limit: int, upper_limit: int, number_points: int) -> np.ndarray:
        """
        Generate a numpy array describing the point density function with a given number of points
        between two limits.

        Parameters
        ----------
        lower_limit
        upper_limit
        number_points

        Returns
        -------
        An array illustrating the point density
        """


class Observable(AbstractObservable):
    def __init__(self, mean: float, deviation: float):
        """
        An observable feature associated with one species which has a Gaussian distribution for that species.

        Parameters
        ----------
        mean
            The mean of the distribution
        deviation
            The standard deviation of the distribution
        """
        self.mean = mean
        self.deviation = deviation

    @property
    def distribution(self):
        """
        A function for sampling a normal distribution
        """
        return stats.norm(loc=self.mean, scale=self.deviation)

    def pdf(
        self, lower_limit: int = -2, upper_limit: int = 2, number_points: int = 1000
    ) -> np.ndarray:
        """
        Generate a numpy array describing the point density function with a given number of points
        between two limits.

        Parameters
        ----------
        lower_limit
        upper_limit
        number_points

        Returns
        -------
        An array illustrating the point density
        """
        return self.distribution.pdf(
            np.linspace(lower_limit, upper_limit, number_points)[:, None]
        )

    def __eq__(self, other):
        return self.mean == other.mean and self.deviation == other.deviation

    def __hash__(self):
        return hash(self.mean) + hash(self.deviation)

    def __add__(self, other):
        return CompoundObservable([1.0, 1.0], [self, other])


class CompoundObservable(AbstractObservable):
    @assert_lengths_match
    def __init__(self, abundances: List[float], observables: List[Observable]):
        """
        Collates observables producing a PDF that sums member PDFs multiplied by their abundances.

        Parameters
        ----------
        abundances
            A list of abundances for the species from which the observable PDFs were taken.
        observables
            A list of observables.
        """
        self.abundances = abundances
        self.observables = observables

    def pdf(
        self, lower_limit: int = -2, upper_limit: int = 2, number_of_points: int = 1000
    ) -> np.ndarray:
        """
        Compute the Point Density Function from the constituent PDFs multiplied
        by their abundances.

        Parameters
        ----------
        lower_limit
        upper_limit
        number_of_points

        Returns
        -------
        An array illustrating the pdf
        """
        pdfs = [
            abundance
            * observable.pdf(
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                number_points=number_of_points,
            )
            for abundance, observable in zip(self.abundances, self.observables)
        ]
        sum_array = np.zeros(pdfs[0].shape)
        for pdf in pdfs:
            sum_array = np.add(sum_array, pdf)
        return sum_array
