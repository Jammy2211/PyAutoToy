import copy
from random import randint
from typing import Set, List

import numpy as np

import autofit as af
from time_series.observable import AbstractObservable, CompoundObservable, Observable

LOWER_LIMIT = 0
UPPER_LIMIT = 20
NUMBER_OF_POINTS = 40

GRANULARITY = 100


def pdf(observable: AbstractObservable) -> np.ndarray:
    return observable.pdf(
        LOWER_LIMIT,
        UPPER_LIMIT,
        NUMBER_OF_POINTS
    )


class Data(af.Dataset):
    @property
    def name(self) -> str:
        return "Observables"

    def __init__(
            self,
            timestep,
            **observables
    ):
        """
        Contains a dictionary mapping named observables to their.

        Parameters
        ----------
        observables
        """
        self.timestep = timestep
        self.observables = observables

    @property
    def observable_names(self) -> Set[str]:
        """
        The names of all the observables
        """
        return set(self.observables.keys())

    def __getitem__(self, observable_name):
        return self.observables[observable_name].pdf(
            LOWER_LIMIT,
            UPPER_LIMIT,
            NUMBER_OF_POINTS
        )

    def __str__(self):
        return str({
            key: value.shape
            for key, value
            in self.observables.items()
        })


def rand_positive(upper_limit):
    return randint(
        0,
        upper_limit * GRANULARITY
    ) / GRANULARITY


def generate_data_at_timesteps(
        number_of_observables: int,
        number_of_species: int,
        timesteps: List[int]
) -> List[Data]:
    """
    Generate data over a

    Parameters
    ----------
    number_of_observables
    number_of_species
    timesteps

    Returns
    -------

    """
    base_data = generate_data(
        number_of_observables,
        number_of_species
    )
    data_list = list()
    for timestep in timesteps:
        data = copy.deepcopy(base_data)
        for compound_observable in data.observables.values():
            compound_observable.abundances = [
                rand_positive(
                    1
                ) for _ in range(number_of_species)
            ]
        data.timestep = timestep
        data_list.append(data)
    return data_list


def generate_data(
        number_of_observables: int,
        number_of_species: int
) -> Data:
    """
    Generate random data for a given number of observables and species.

    Parameters
    ----------
    number_of_observables
    number_of_species

    Returns
    -------
    Randomly generated observable distribution data
    """
    compound_observables = dict()
    for number in range(number_of_observables):
        compound_observables[
            str(number)
        ] = CompoundObservable(
            abundances=[
                rand_positive(
                    1
                ) for _ in range(number_of_species)
            ],
            observables=[
                Observable(
                    mean=rand_positive(3),
                    deviation=rand_positive(2)
                ) for _ in range(number_of_species)
            ]
        )
    return Data(
        timestep=0,
        **compound_observables
    )
