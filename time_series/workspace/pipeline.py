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


def make_phase(
        number=None,
        previous_phase=None
):
    # This is our model. It's an object that can be given a unit vector
    # of length number of dimensions to create an instance.
    model = af.ModelMapper()

    # We create a dimension for the abundance of each species at this
    # time step.
    model.abundances = [
        af.UniformPrior(
            lower_limit=0,
            upper_limit=1
        )
        for _ in range(
            NUMBER_OF_SPECIES
        )
    ]

    # If there was a previous phase we use the results of that phase to constrain
    # the species observables in this phase.
    if previous_phase is not None:
        previous_model = previous_phase.result.model
        model.species = previous_model.species

    # We also create a model for each species. We fix the growth rate
    # as we're just fitting for the abundances and observables. Each
    # species has a sub model representing each observable.
    model.species = [
        af.PriorModel(
            ts.Species,
            observables={
                str(number): af.PriorModel(
                    ts.Observable,
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
    phase = ts.TimeSeriesPhase(
        phase_name=f"observation_phase",
        analysis_class=ts.Analysis,
        model=model,
        data_index=number
    )

    # The phase uses MultiNest by default. We can actually change that if we
    # want in the constructor. We can also fiddle with its settings.
    phase.optimizer.const_efficiency_mode = True
    phase.optimizer.n_live_points = 20
    phase.optimizer.sampling_efficiency = 0.8

    return phase


def make_pipeline(timesteps=1):
    phase = None
    phases = list()
    for timestep in range(timesteps):
        phase = make_phase(
            number=timestep,
            previous_phase=phase
        )
        phases.append(phase)
    # We stick our phases into a pipeline.
    return af.Pipeline(
        "timeseries",
        *phases
    )


if __name__ == "__main__":
    pipeline = make_pipeline()
    data = ts.generate_data_at_timesteps(
        number_of_observables=NUMBER_OF_OBSERVABLES,
        number_of_species=NUMBER_OF_SPECIES,
        timesteps=[0]
    )
    pipeline.run(
        data
    )
