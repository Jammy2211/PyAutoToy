import autofit as af
from time_series.data import pdf, Data, TimeSeriesData
from time_series.fit import SingleTimeFit
from time_series.lotka_voltera import LotkaVolteraModel
from time_series.species import SpeciesObservables


class TimeSeriesAnalysis(af.Analysis):
    def __init__(self, dataset: TimeSeriesData):
        self.dataset = dataset

    def fit(self, instance: af.ModelInstance) -> float:
        """
        Grow the population forwards in time and compare the observables
        to observation at given points in time.

        Parameters
        ----------
        instance
            With a list of abundances and a species_collection object

        Returns
        -------
        How well the model matches observations.
        """
        initial_abundances = instance.abundances
        species_collection = instance.species_collection

        lotka_voltera = LotkaVolteraModel(
            species_collection
        )

        abundances = initial_abundances
        time = 0
        fitness = 0

        for data_time, dataset in self.dataset:
            while time < data_time:
                abundances = lotka_voltera.step(
                    abundances
                )
                time += 1

            species_observables = SpeciesObservables(
                abundances=abundances,
                species=species_collection
            )
            for observable_name in dataset.observable_names:
                fitness -= SingleTimeFit(
                    self.dataset[observable_name],
                    pdf(species_observables[observable_name])
                ).chi_squared
        return fitness

    def visualize(self, instance, during_analysis):
        pass


class SingleTimeAnalysis(af.Analysis):

    def __init__(self, dataset: Data):
        """
        Used to compare model instances to the data.

        Parameters
        ----------
        dataset
            Data comprising observations taken at a particular point in time.
        """
        self.dataset = dataset

    def fit(self, instance: af.ModelInstance) -> float:
        """
        Determine how well a set of species with given distributions
        for a set of observables and given abundances fits an observed
        set of observables.

        Parameters
        ----------
        instance
            An instance of a model

        Returns
        -------
        The evidence for the model
        """
        fitness = 0
        species_observables = SpeciesObservables(
            abundances=instance.abundances,
            species=instance.species
        )
        for observable_name in self.dataset.observable_names:
            fitness -= SingleTimeFit(
                self.dataset[observable_name],
                pdf(species_observables[observable_name])
            ).chi_squared
        return fitness

    def visualize(self, instance, during_analysis):
        pass
