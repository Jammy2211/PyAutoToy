import os
from os import path

import pytest

import autofit as af
import toy_gaussian as toy
import autoarray as aa
from toy_gaussian.test.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/config/phase_imaging_7x7".format(directory)
    )


def clean_images():
    try:
        os.remove("{}/source_lens_phase/source_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/lens_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/model_image_0.fits".format(directory))
    except FileNotFoundError:
        pass
    af.conf.instance.dataset_path = directory


class TestPhase:
    def test__make_analysis__mask_input_uses_mask__no_mask_uses_mask_function(
        self, phase_imaging_7x7, imaging_7x7
    ):
        # If an input mask is supplied and there is no mask function, we use mask input.

        mask_input = aa.mask.circular(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=1.0, sub_size=1, radius=1.5
        )

        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_input)

        assert (analysis.masked_imaging.mask == mask_input).all()

    def test__make_analysis__mask_changes_sub_size_depending_on_phase_attribute(
        self, phase_imaging_7x7, imaging_7x7
    ):
        # If an input mask is supplied and there is no mask function, we use mask input.

        mask_input = aa.mask.circular(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=1, sub_size=1, radius=1.5
        )

        phase_imaging_7x7.meta_imaging_fit.sub_size = 1
        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_input)

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 1

        phase_imaging_7x7.meta_imaging_fit.sub_size = 2
        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_input)

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 2


class TestResult:
    def test__results_of_phase_are_available_as_properties(self, imaging_7x7, mask_7x7):
        clean_images()

        phase_imaging_7x7 = toy.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            gaussians=[toy.SphericalGaussian],
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7)

        assert isinstance(result, toy.AbstractPhase.Result)

    def test__results_of_phase_include_mask__available_as_property(
        self, imaging_7x7, mask_7x7
    ):
        clean_images()

        phase_imaging_7x7 = toy.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            gaussians=[toy.SphericalGaussian],
            sub_size=2,
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7)

        assert (result.mask == mask_7x7).all()


class TestPhasePickle:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_7x7, mask_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.MockAnalysis(value=1)

        phase_imaging_7x7 = toy.PhaseImaging(
            phase_name="phase_name",
            optimizer_class=mock_pipeline.MockNLO,
            gaussians=[toy.SphericalGaussian],
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(dataset=imaging_7x7, results=None, mask=mask_7x7)
        assert result is not None

        phase_imaging_7x7 = toy.PhaseImaging(
            phase_name="phase_name",
            optimizer_class=mock_pipeline.MockNLO,
            gaussians=[toy.SphericalGaussian],
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(dataset=imaging_7x7, results=None, mask=mask_7x7)
        assert result is not None

        class CustomPhase(toy.PhaseImaging):
            def customize_priors(self, results):
                self.gaussians[0] = toy.SphericalGaussian()

        phase_imaging_7x7 = CustomPhase(
            phase_name="phase_name",
            optimizer_class=mock_pipeline.MockNLO,
            gaussians=[toy.SphericalGaussian],
        )
        phase_imaging_7x7.make_analysis = make_analysis
