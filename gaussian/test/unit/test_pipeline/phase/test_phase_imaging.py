import os
from os import path

import numpy as np
import pytest

import autofit as af
import gaussian as g
import autoarray as aa
from autoarray.fit.fit import fit_masked_dataset
from gaussian.test.mock import mock_pipeline

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


class TestPhase(object):
    def test__make_analysis__masks_image_and_noise_map_correctly(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)

        assert (
            analysis.masked_imaging.image.in_2d
            == imaging_7x7.image.in_2d * np.invert(mask_7x7)
        ).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == imaging_7x7.noise_map.in_2d * np.invert(mask_7x7)
        ).all()

    def test__make_analysis__phase_info_is_made(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)

        file_phase_info = "{}/{}".format(
            phase_imaging_7x7.optimizer.paths.phase_output_path, "phase.info"
        )

        phase_info = open(file_phase_info, "r")

        optimizer = phase_info.readline()
        sub_size = phase_info.readline()

        phase_info.close()

        assert optimizer == "Optimizer = MockNLO \n"
        assert sub_size == "Sub-grid size = 2 \n"

    def test__fit_using_imaging(self, imaging_7x7, mask_7x7):
        clean_images()

        phase_imaging_7x7 = g.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            gaussians=[g.SphericalGaussian, g.SphericalGaussian],
            phase_name="test_phase_test_fit",
        )

        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7)
        assert isinstance(result.instance.gaussians[0], g.SphericalGaussian)
        assert isinstance(result.instance.gaussians[1], g.SphericalGaussian)

    def test_modify_image(self, imaging_7x7, mask_7x7):
        class MyPhase(g.PhaseImaging):
            def modify_image(self, image, results):
                assert imaging_7x7.image.shape_2d == image.shape_2d
                image = aa.array.full(
                    fill_value=20.0, shape_2d=(7, 7), pixel_scales=image.pixel_scales
                )
                return image

        phase_imaging_7x7 = MyPhase(phase_name="phase_imaging_7x7")

        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)
        assert (
            analysis.masked_imaging.image.in_2d
            == 20.0 * np.ones(shape=(7, 7)) * np.invert(mask_7x7)
        ).all()
        assert (analysis.masked_imaging.image.in_1d == 20.0 * np.ones(shape=9)).all()

    def test__masked_imaging_signal_to_noise_limit(self, imaging_7x7, mask_7x7_1_pix):
        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=1.0
        )

        phase_imaging_7x7 = g.PhaseImaging(
            phase_name="phase_imaging_7x7", signal_to_noise_limit=1.0
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix
        )
        assert (
            analysis.masked_imaging.image.in_2d
            == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
        ).all()

        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=0.1
        )

        phase_imaging_7x7 = g.PhaseImaging(
            phase_name="phase_imaging_7x7", signal_to_noise_limit=0.1
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix
        )
        assert (
            analysis.masked_imaging.image.in_2d
            == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
        ).all()

    def test__masked_imaging_is_binned_up(self, imaging_7x7, mask_7x7_1_pix):
        binned_up_imaging = imaging_7x7.binned_from_bin_up_factor(bin_up_factor=2)

        binned_up_mask = mask_7x7_1_pix.mapping.binned_mask_from_bin_up_factor(
            bin_up_factor=2
        )

        phase_imaging_7x7 = g.PhaseImaging(
            phase_name="phase_imaging_7x7", bin_up_factor=2
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix
        )
        assert (
            analysis.masked_imaging.image.in_2d
            == binned_up_imaging.image.in_2d * np.invert(binned_up_mask)
        ).all()
        assert (analysis.masked_imaging.psf == binned_up_imaging.psf).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == binned_up_imaging.noise_map.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (analysis.masked_imaging.mask == binned_up_mask).all()

        masked_imaging = aa.masked.imaging(imaging=imaging_7x7, mask=mask_7x7_1_pix)

        binned_up_masked_imaging = masked_imaging.binned_from_bin_up_factor(
            bin_up_factor=2
        )

        assert (
            analysis.masked_imaging.image.in_2d
            == binned_up_masked_imaging.image.in_2d * np.invert(binned_up_mask)
        ).all()
        assert (analysis.masked_imaging.psf == binned_up_masked_imaging.psf).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == binned_up_masked_imaging.noise_map.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (analysis.masked_imaging.mask == binned_up_masked_imaging.mask).all()

        assert (
            analysis.masked_imaging.image.in_1d == binned_up_masked_imaging.image.in_1d
        ).all()
        assert (
            analysis.masked_imaging.noise_map.in_1d
            == binned_up_masked_imaging.noise_map.in_1d
        ).all()

    def test__fit_figure_of_merit__matches_correct_fit_given_gaussian_profiles(
        self, imaging_7x7, mask_7x7
    ):
        gaussian = g.SphericalGaussian(intensity=0.1)

        phase_imaging_7x7 = g.PhaseImaging(
            gaussians=[gaussian], sub_size=2, phase_name="test_phase"
        )

        analysis = phase_imaging_7x7.make_analysis(dataset=imaging_7x7, mask=mask_7x7)
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase_imaging_7x7.meta_imaging_fit.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        masked_imaging = aa.masked.imaging(imaging=imaging_7x7, mask=mask)

        model_image = gaussian.profile_image_from_grid(grid=masked_imaging.grid)
        fit = fit_masked_dataset(
            masked_dataset=masked_imaging, model_data=model_image.in_1d_binned
        )

        assert fit.likelihood == fit_figure_of_merit
