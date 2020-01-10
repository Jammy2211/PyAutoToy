from time_series.species import SpeciesCollection


class LotkaVolteraModel:
    def __init__(
            self,
            species_collection: SpeciesCollection,
            capacity: float = 1.0
    ):
        self.species_collection = species_collection
        self.capacity = capacity
