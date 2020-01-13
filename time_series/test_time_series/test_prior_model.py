import itertools

import pytest

import autofit as af
from time_series import matrix_prior_model as m
from time_series import species as s


@pytest.fixture(autouse=True)
def reset_model_id():
    af.ModelObject._ids = itertools.count()


@pytest.fixture(name="matrix_prior_model")
def make_matrix_prior_model():
    return m.MatrixPriorModel(
        s.SpeciesCollection,
        [
            s.Species,
            s.Species
        ]
    )


@pytest.fixture(name="self_interacting_prior_model")
def make_self_interacting_prior_model(matrix_prior_model):
    prior = af.UniformPrior()
    for i in range(len(matrix_prior_model)):
        matrix_prior_model[i, i] = prior
    return matrix_prior_model


class TestBasicBehaviour:
    def test_instantiate(self, matrix_prior_model):
        assert isinstance(
            matrix_prior_model,
            af.CollectionPriorModel
        )
        assert matrix_prior_model.prior_count == 2

        instance = matrix_prior_model.instance_from_prior_medians()
        assert isinstance(instance, s.SpeciesCollection)
        assert len(instance) == 2

    def test_priors(self, self_interacting_prior_model):
        assert self_interacting_prior_model.prior_count == 3

        instance = self_interacting_prior_model.instance_from_prior_medians()
        assert instance[0, 0] == 0.5
        assert instance[1, 1] == 0.5
        assert instance[0, 1] == 0.0
        assert instance[1, 0] == 0.0

    def test_model_info(self, self_interacting_prior_model):
        assert self_interacting_prior_model.info == """0
    growth_rate                                                                           UniformPrior, lower_limit = 0.0, upper_limit = 1.0
    interactions
        SpeciesPriorModel 0                                                               UniformPrior, lower_limit = 0.0, upper_limit = 1.0
1
    growth_rate                                                                           UniformPrior, lower_limit = 0.0, upper_limit = 1.0
    interactions
        SpeciesPriorModel 3                                                               UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""
