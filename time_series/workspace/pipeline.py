from os import path
from typing import List

import autofit as af
import time_series as ts

NUMBER_OF_SPECIES = 3
NUMBER_OF_OBSERVABLES = 2
TIMESTEPS = [7, 21, 101]

directory = path.dirname(path.realpath(__file__))

af.conf.instance = af.conf.Config(
    path.join(directory, "config"),
    path.join(directory, "output")
)


def make_abundances():
    return [
        af.UniformPrior(
            lower_limit=0,
            upper_limit=1
        )
        for _ in range(
            NUMBER_OF_SPECIES
        )
    ]


def make_phase(
        number=None,
        previous_phase=None
):
    # This is our model. It's an object that can be given a unit vector
    # of length number of dimensions to create an instance.
    model = af.ModelMapper()

    # We create a dimension for the abundance of each species at this
    # time step.
    model.abundances = make_abundances()

    # If there was a previous phase we use the results of that phase to constrain
    # the species observables in this phase.
    if previous_phase is not None:
        previous_model = previous_phase.result.model
        model.species = previous_model.species
    else:
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
    phase = ts.SingleTimePhase(
        phase_name=f"observation_phase",
        analysis_class=ts.SingleTimeAnalysis,
        model=model,
        data_index=number
    )

    # The phase uses MultiNest by default. We can actually change that if we
    # want in the constructor. We can also fiddle with its settings.
    phase.optimizer.const_efficiency_mode = True
    phase.optimizer.n_live_points = 20
    phase.optimizer.sampling_efficiency = 0.8

    return phase


def make_pipeline(timesteps: List[int]) -> af.Pipeline:
    """
    Create a pipeline that fits a time series of observations.

    - First fit observations at each point in time with the abundance
    of each species and the coefficients of the distribution of each
    observable with respect to each species for each point in time.
    - For each of these phases use priors on the observables from the
    previous phase.
    - Finally, fit observations at every time point, fixing the
    observable distribution parameters but varying the initial abundance,
    growth rate and interactions of each species. In this phase the
    Lotka Voltera model is used to compute the growth given each
    parameterisation.

    Parameters
    ----------
    timesteps
        A list of timesteps for which observations are provided.

    Returns
    -------
    A pipeline comprising a phase for each timestep and a final
    phase for a full time series.
    """

    # Loop through the timesteps and make a phase for each. Use
    # the results from the previous phase to constrain the next
    # phase if a previous phase exists.
    phase = None
    single_timestep_phases: List[af.Phase] = list()
    for timestep in timesteps:
        phase = make_phase(
            number=timestep,
            previous_phase=phase
        )
        single_timestep_phases.append(phase)

    model = af.ModelMapper()
    model.abundances = make_abundances()

    instance_species_list = phase.result.instance.species

    matrix = ts.MatrixPriorModel(
        ts.SpeciesCollection,
        items=[
            ts.SpeciesPriorModel(
                cls=ts.Species,
                observables=species.observables
            )
            for species in instance_species_list
        ]
    )
    for i in range(NUMBER_OF_SPECIES):
        for j in range(NUMBER_OF_SPECIES):
            matrix[i, j] = af.UniformPrior(0.0, 1.0)

    model.species_collection = matrix

    time_series_phase = af.Phase(
        phase_name="time_series_phase",
        model=model,
        analysis_class=ts.TimeSeriesAnalysis
    )

    # We stick our phases into a pipeline.
    return af.Pipeline(
        "timeseries",
        *(single_timestep_phases + [time_series_phase])
    )


if __name__ == "__main__":
    pipeline = make_pipeline(
        timesteps=TIMESTEPS
    )
    data = ts.generate_data_at_timesteps(
        number_of_observables=NUMBER_OF_OBSERVABLES,
        number_of_species=NUMBER_OF_SPECIES,
        timesteps=TIMESTEPS
    )
    pipeline.run(
        data,
    )
