import itertools

import pytest

import autofit as af
import time_series as ts


@pytest.fixture(autouse=True)
def reset_model_id():
    af.ModelObject._ids = itertools.count()


@pytest.fixture(name="matrix_prior_model")
def make_matrix_prior_model():
    return ts.MatrixPriorModel(
        ts.SpeciesCollection,
        [
            ts.SpeciesPriorModel(ts.Species),
            ts.SpeciesPriorModel(ts.Species)
        ]
    )


@pytest.fixture(name="self_interacting_prior_model")
def make_self_interacting_prior_model(matrix_prior_model):
    prior = af.UniformPrior()
    for i in range(len(matrix_prior_model)):
        matrix_prior_model[i, i] = prior
    return matrix_prior_model


def test_observables():
    model = af.PriorModel(
        ts.Species,
        observables=af.CollectionPriorModel(
            one=ts.Observable,
            two=ts.Observable
        )
    )

    assert model.prior_count == 5

    instance = model.instance_from_prior_medians()
    assert isinstance(
        instance.observables["one"],
        ts.Observable
    )


class TestBasicBehaviour:
    def test_instantiate(self, matrix_prior_model):
        assert isinstance(
            matrix_prior_model,
            af.CollectionPriorModel
        )
        assert matrix_prior_model.prior_count == 2

        instance = matrix_prior_model.instance_from_prior_medians()
        assert isinstance(instance, ts.SpeciesCollection)
        assert len(instance) == 2

    def test_priors(self, self_interacting_prior_model):
        assert self_interacting_prior_model.prior_count == 3

        instance = self_interacting_prior_model.instance_from_prior_medians()
        assert instance[0, 0] == 0.5
        assert instance[1, 1] == 0.5
        assert instance[0, 1] == 0.0
        assert instance[1, 0] == 0.0

    def test_mix(self, self_interacting_prior_model):
        self_interacting_prior_model[0, 1] = 1.0
        self_interacting_prior_model[1, 0] = 2.0

        instance = self_interacting_prior_model.instance_from_prior_medians()
        assert instance[0, 0] == 0.5
        assert instance[1, 1] == 0.5
        assert instance[0, 1] == 1.0
        assert instance[1, 0] == 2.0

    def test_model_info(self, self_interacting_prior_model):
        assert self_interacting_prior_model.info == """0
    growth_rate                                                                           UniformPrior, lower_limit = 0.0, upper_limit = 1.0
    interactions
        SpeciesPriorModel 0                                                               UniformPrior, lower_limit = 0.0, upper_limit = 1.0
1
    growth_rate                                                                           UniformPrior, lower_limit = 0.0, upper_limit = 1.0
    interactions
        SpeciesPriorModel 3                                                               UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""
