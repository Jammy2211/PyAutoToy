import builtins
import pytest

import autofit as af
import gaussian as g
from autofit import Paths


class MockAnalysis:
    def __init__(self, shape, value):
        self.shape = shape
        self.value = value


class MockMask:
    pass


class Optimizer:
    def __init__(self, phase_name="dummy_phase"):
        self.phase_name = phase_name
        self.phase_path = ""


class DummyPhaseImaging(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name, phase_tag=""):
        super().__init__(Paths(phase_name=phase_name, phase_tag=phase_tag))
        self.dataset = None
        self.results = None
        self.mask = None

        self.optimizer = Optimizer(phase_name)

    def run(self, dataset, results, mask=None):
        self.dataset = dataset
        self.results = results
        self.mask = mask
        self.assert_and_save_pickle()
        return af.Result(af.ModelInstance(), 1)


class MockImagingData:
    pass


class MockFile:
    def __init__(self):
        self.text = None
        self.filename = None

    def write(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(name="mock_files", autouse=True)
def make_mock_file(monkeypatch):
    files = []

    def mock_open(filename, flag, *args, **kwargs):
        assert flag in ("w+", "w+b", "a")
        file = MockFile()
        file.filename = filename
        files.append(file)
        return file

    monkeypatch.setattr(builtins, "open", mock_open)
    yield files


class TestMetaData:
    def test_files(self, mock_files):
        pipeline = g.PipelineDataset(
            "pipeline_name", DummyPhaseImaging(phase_name="phase_name")
        )
        pipeline.run(dataset=MockImagingData(), mask=MockMask(), data_name="data_name")

        print(mock_files[1].text)

        assert (
            "pipeline=pipeline_name\nphase=phase_name\ndataset_name=data_name"
            in mock_files[1].text
        )

        assert "phase_name///non_linear.pickle" in mock_files[2].filename


class TestPassMask:
    def test_pass_mask(self):
        mask = MockMask()
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        pipeline = g.PipelineDataset("", phase_1, phase_2)
        pipeline.run(dataset=MockImagingData(), mask=mask)

        assert phase_1.mask is mask
        assert phase_2.mask is mask


class TestPipelineImaging:
    def test_run_pipeline(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")

        pipeline = g.PipelineDataset("", phase_1, phase_2)

        pipeline.run(dataset=MockImagingData(), mask=MockMask())

        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhaseImaging("one")
        phase_2 = DummyPhaseImaging("two")
        phase_3 = DummyPhaseImaging("three")

        pipeline1 = g.PipelineDataset("", phase_1, phase_2)
        pipeline2 = g.PipelineDataset("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
