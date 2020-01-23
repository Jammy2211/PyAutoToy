from os import path

import autofit as af
import time_series as ts

NUMBER_OF_SPECIES = 3
NUMBER_OF_OBSERVABLES = 2

directory = path.dirname(path.realpath(__file__))

af.conf.instance = af.conf.Config(
    path.join(directory, "config"),
    path.join(directory, "output")
)


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
        af.PriorModel(
            ts.Species,
            observables={
                str(number): af.PriorModel(
                    ts.Observable,
                    mean=af.GaussianPrior(3, 1),
                    deviation=af.GaussianPrior(2, 1)
                )
                for number in range(NUMBER_OF_OBSERVABLES)
            },
            growth_rate=1.0
        )
        for _ in range(NUMBER_OF_SPECIES)
    ]

    phase = af.Phase(
        phase_name="phase_1",
        analysis_class=ts.Analysis,
        model=model
    )

    phase.optimizer.const_efficiency_mode = True
    phase.optimizer.n_live_points = 20
    phase.optimizer.sampling_efficiency = 0.8

    result = phase.run(
        data
    )

    print(result.instance)


if __name__ == "__main__":
    run_phase()
