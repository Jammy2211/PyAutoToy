import autofit as af
from time_series.fit import Fit
from time_series.util import pdf


class Analysis(af.Analysis):
    def fit(self, instance):
        fitness = 0
        species_observables = instance.species_observables
        for observable_name in self.data.observable_names:
            fitness -= Fit(
                self.data[observable_name],
                pdf(species_observables[observable_name])
            ).chi_squared
        return fitness

    def visualize(self, instance, during_analysis):
        pass

    def __init__(self, data):
        self.data = data
