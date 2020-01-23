from random import randint

import autofit as af
import time_series as ts

LOWER_LIMIT = 0
UPPER_LIMIT = 20
NUMBER_OF_POINTS = 40

GRANULARITY = 100


def rand_positive(upper_limit):
    return randint(
        0,
        upper_limit * GRANULARITY
    ) / GRANULARITY


def generate_data(
        number_of_observables,
        number_of_species
):
    compound_observables = dict()
    for number in range(number_of_observables):
        compound_observables[
            str(number)
        ] = ts.CompoundObservable(
            abundances=[
                rand_positive(
                    1
                ) for _ in range(number_of_species)
            ],
            observables=[
                ts.Observable(
                    mean=rand_positive(3),
                    deviation=rand_positive(2)
                ) for _ in range(number_of_species)
            ]
        ).pdf(
            LOWER_LIMIT,
            UPPER_LIMIT,
            NUMBER_OF_POINTS
        )
    return ts.Data(
        **compound_observables
    )


def pdf(observable):
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
            **observables
    ):
        self.observables = observables

    @property
    def observable_names(self):
        return set(self.observables.keys())

    def __getitem__(self, observable_name):
        return self.observables[observable_name]

    def __str__(self):
        return str({
            key: value.shape
            for key, value
            in self.observables.items()
        })
