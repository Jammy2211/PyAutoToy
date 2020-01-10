import numpy as np

from time_series import species as s


class TestSpecies:
    def test_interaction(self, species_a):
        assert species_a.interactions[species_a] == 1.0

        species_a.interactions[species_a] = 0.5
        assert species_a.interactions[species_a] == 0.5


class TestSpeciesCollection:
    def test_trivial_interaction_matrix(self, species_a):
        collection = s.SpeciesCollection(species_a)

        assert collection.species == [species_a]
        assert collection.interaction_matrix == np.array([[1.0]])

    def test_interaction_matrix(
            self,
            species_a,
            species_b,
            species_collection
    ):
        species_a.interactions[species_b] = 0.5
        species_b.interactions[species_a] = 0.7

        assert (species_collection.interaction_matrix == np.array(
            [
                [1.0, 0.5],
                [0.7, 1.0]
            ]
        )).all()

    def test_growth_rate_vector(
            self,
            species_collection
    ):
        assert (species_collection.growth_rate_vector == np.array(
            [1.0, 2.0]
        )).all()

    def test_simple_indexing(
            self,
            species_a,
            species_b,
            species_collection
    ):
        assert species_collection[0] == species_a
        assert species_collection[1] == species_b

        species_collection[0] = species_b
        species_collection[1] = species_a

        assert species_collection[0] == species_b
        assert species_collection[1] == species_a

    def test_interaction_indexing(
            self,
            species_a,
            species_b,
            species_collection
    ):
        species_collection[0, 0] = 0.0
        species_collection[1, 0] = 1.0
        species_collection[0, 1] = 0.1
        species_collection[1, 1] = 1.1

        assert species_a[species_a] == 0.0
        assert species_b[species_a] == 1.0
        assert species_a[species_b] == 0.1
        assert species_b[species_b] == 1.1

        assert species_collection[0, 0] == 0.0
        assert species_collection[1, 0] == 1.0
        assert species_collection[0, 1] == 0.1
        assert species_collection[1, 1] == 1.1
