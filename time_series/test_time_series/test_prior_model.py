import pytest

import autofit as af
from time_series import matrix_prior_model as m
from time_series import species as s


@pytest.fixture(name="matrix_prior_model")
def make_matrix_prior_model():
    return m.MatrixPriorModel(
        s.SpeciesCollection,
        [
            s.Species,
            s.Species
        ]
    )


class TestBasicBehaviour:
    def test_instantiate(self, matrix_prior_model):
        assert isinstance(
            matrix_prior_model,
            af.CollectionPriorModel
        )
        assert matrix_prior_model.prior_count == 0

        instance = matrix_prior_model.instance_from_prior_medians()
        assert isinstance(instance, s.SpeciesCollection)
        assert len(instance) == 2

    def test_priors(self, matrix_prior_model):
        prior = af.UniformPrior()
        for i in range(len(matrix_prior_model)):
            matrix_prior_model[i, i] = prior

        assert matrix_prior_model.prior_count == 1

        instance = matrix_prior_model.instance_from_prior_medians()
        assert instance[0, 0] == 0.5
        assert instance[1, 1] == 0.5
        assert instance[0, 1] == 0.0
        assert instance[1, 0] == 0.0
