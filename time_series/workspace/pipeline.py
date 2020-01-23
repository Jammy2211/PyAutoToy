import autofit as af
import time_series as ts

NUMBER_OF_SPECIES = 5
NUMBER_OF_OBSERVABLES = 3


def run_phase():
    model = af.ModelMapper()
    data = ts.generate_data(
        number_of_observables=NUMBER_OF_OBSERVABLES,
        number_of_species=NUMBER_OF_SPECIES
    )

    phase = af.Phase(
        analysis_class=ts.Analysis,
        model=model
    )
    result = phase.run(
        data
    )

    print(result.instance)


if __name__ == "__main__":
    pass
