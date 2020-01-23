from random import randint

import autofit as af
import time_series as ts

NUMBER_OF_SPECIES = 5
NUMBER_OF_OBSERVABLES = 3

LOWER_LIMIT = 0
UPPER_LIMIT = 20
NUMBER_OF_POINTS = 400

GRANULARITY = 100


def rand_positive(upper_limit):
    return randint(
        0,
        upper_limit * GRANULARITY
    ) / GRANULARITY


def generate_data():
    compound_observables = dict()
    for number in range(NUMBER_OF_OBSERVABLES):
        compound_observables[
            str(number)
        ] = ts.CompoundObservable(
            abundances=[
                rand_positive(
                    1
                ) for _ in range(NUMBER_OF_SPECIES)
            ],
            observables=[
                ts.Observable(
                    mean=rand_positive(3),
                    deviation=rand_positive(2)
                ) for _ in range(NUMBER_OF_SPECIES)
            ]
        ).pdf(
            LOWER_LIMIT,
            UPPER_LIMIT,
            NUMBER_OF_POINTS
        )
    return ts.Data(
        **compound_observables
    )


def run_phase():
    model = af.ModelMapper()
    data = ts.Data()

    phase = af.Phase(
        analysis_class=ts.Analysis,
        model=model
    )
    result = phase.run(
        data
    )

    print(result.instance)


if __name__ == "__main__":
    print(generate_data())
