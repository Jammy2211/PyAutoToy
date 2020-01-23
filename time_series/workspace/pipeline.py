import json

import autofit as af
import time_series as ts

NUMBER_OF_SPECIES = 5
NUMBER_OF_OBSERVABLES = 3


def run_phase():
    data = ts.generate_data(
        number_of_observables=NUMBER_OF_OBSERVABLES,
        number_of_species=NUMBER_OF_SPECIES
    )

    model = af.ModelMapper()
    model.abundances = [
        af.UniformPrior(
            0, 1
        )
        for _ in range(
            NUMBER_OF_SPECIES
        )
    ]
    model.species = [
        ts.Species(
            observables={
                str(number): af.PriorModel(
                    ts.Observable,
                    mean=af.GaussianPrior(3, 1),
                    deviation=af.GaussianPrior(2, 1)
                )
                for number in range(NUMBER_OF_OBSERVABLES)
            }
        )
        for _ in range(NUMBER_OF_SPECIES)
    ]

    # 5 * 3 * 2 priors for observables + 5 priors for abundances
    assert model.prior_count == 35

    phase = af.Phase(
        phase_name="Species",
        analysis_class=ts.Analysis,
        model=model
    )
    result = phase.run(
        data
    )

    print(result.instance)


if __name__ == "__main__":
    run_phase()
