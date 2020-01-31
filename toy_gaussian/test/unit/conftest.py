import autofit as af
import toy_gaussian as toy
from autoarray.mask import mask
from test_autoarray.unit.conftest import *
from toy_gaussian.test.mock import mock_pipeline

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(
        path.join(directory, "test_files/config"), path.join(directory, "output")
    )


#
# MODEL #
#


@pytest.fixture(name="gaussian_0")
def make_gaussian_0():
    # noinspection PyTypeChecker
    return toy.SphericalGaussian(centre=(0.0, 0.0), intensity=1.0, sigma=0.5)


@pytest.fixture(name="gaussian_1")
def make_gaussian_1():
    # noinspection PyTypeChecker
    return toy.SphericalGaussian(centre=(0.5, 0.5), intensity=2.0, sigma=1.0)


@pytest.fixture(name="gaussians")
def make_gaussians(gaussian_0, gaussian_1):
    # noinspection PyTypeChecker
    return [gaussian_0, gaussian_1]


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data():
    return toy.PhaseDataset(
        optimizer_class=mock_pipeline.MockNLO, phase_tag="", phase_name="test_phase"
    )


@pytest.fixture(name="phase_imaging_7x7")
def make_phase_imaging_7x7():
    return toy.PhaseImaging(
        optimizer_class=mock_pipeline.MockNLO, phase_name="test_phase"
    )


@pytest.fixture(name="results_7x7")
def make_results(mask_7x7,):
    return mock_pipeline.MockResults(mask=mask_7x7)


@pytest.fixture(name="results_collection_7x7")
def make_results_collection(results_7x7):
    results_collection = af.ResultsCollection()
    results_collection.add("phase", results_7x7)
    return results_collection


@pytest.fixture(name="include_all")
def make_include_all():
    return toy.plot.Include(
        origin=True,
        mask=True,
        grid=True,
        border=True,
        positions=True,
        gaussian_centres=True,
        inversion_pixelization_grid=True,
        inversion_grid=True,
        inversion_border=True,
        inversion_image_pixelization_grid=True,
    )
