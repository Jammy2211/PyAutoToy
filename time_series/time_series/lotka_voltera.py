from time_series.species import SpeciesCollection


class LotkaVolteraModel:
    def __init__(
            self,
            species_collection: SpeciesCollection,
            capacity: float = 1.0
    ):
        self.species_collection = species_collection
        self.capacity = capacity

    def growth_rates(self, population):
        return 1 - self.species_collection.interaction_matrix.dot(
            population
        ) / self.capacity

    def change(self, population):
        population_coefficients = population * self.species_collection.growth_rate_vector
        return population_coefficients * self.growth_rates(population)

    def step(self, population):
        return population + self.change(population)
