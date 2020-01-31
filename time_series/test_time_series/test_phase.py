import pytest

import autofit as af
import time_series as ts


class MockAnalysis(af.Analysis):
    def __init__(self, data):
        self.data = data

    def fit(self, instance):
        pass

    def visualize(self, instance, during_analysis):
        pass


@pytest.fixture(name="phase")
def make_phase():
    return ts.TimeSeriesPhase(
        "phase",
        analysis_class=MockAnalysis,
        data_index=0
    )


def test_name(phase):
    assert phase.phase_name == "phase_0"


def test_data(phase):
    analysis = phase.make_analysis([0, 1, 2, 3])
    assert analysis.data == 0
