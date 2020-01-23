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


def make_pipeline():
    # This is our model. It's an object that can be given a unit vector
    # of length number of dimensions to create an instance.
    model = af.ModelMapper()

    # We create a dimension for the abundance of each species at this
    # time step.
    model.abundances = [
        af.UniformPrior(
            0, 1
        )
        for _ in range(
            NUMBER_OF_SPECIES
        )
    ]
    # We also create a model for each species. We fix the growth rate
    # as we're just fitting for the abundances and observables. Each
    # species has a sub model representing each observable.
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

    # Next we create a phase. The phase comprises the model and an analysis
    # class. An analysis is instantiated with a data object and has a function
    # that evaluates the fit of any given instance.
    # In this case, the Analysis expects the instance to have a list of
    # abundances and a list of species.
    phase = af.Phase(
        phase_name="observation_phase",
        analysis_class=ts.Analysis,
        model=model
    )

    # The phase uses MultiNest by default. We can actually change that if we
    # want in the constructor. We can also fiddle with its settings.
    phase.optimizer.const_efficiency_mode = True
    phase.optimizer.n_live_points = 20
    phase.optimizer.sampling_efficiency = 0.8

    # We stick our phases into a pipeline.
    return af.Pipeline(
        "timeseries",
        phase
    )


if __name__ == "__main__":
    pipeline = make_pipeline()
    data = ts.generate_data(
        number_of_observables=NUMBER_OF_OBSERVABLES,
        number_of_species=NUMBER_OF_SPECIES
    )
    pipeline.run(
        data
    )
