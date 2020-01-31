import autofit as af
from time_series.fit import SingleTimeFit
from time_series.data import pdf, Data, TimeSeriesData
from time_series.species import SpeciesObservables


class TimeSeriesAnalysis(af.Analysis):
    def __init__(self, dataset: TimeSeriesData):
        self.dataset = dataset

    def fit(self, instance):
        pass

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
