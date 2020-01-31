import autoarray as aa
from toy_gaussian.src.pipeline import visualizer as vis
import os
import pytest
from os import path
import shutil
from autofit import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return "{}/../test_files/plotting/visualizer/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


class TestPhaseImagingVisualizer:
    def test__visualizes_imaging_using_configs(
        self, masked_imaging_7x7, include_all, plot_path, plot_patch
    ):

        visualizer = vis.PhaseImagingVisualizer(
            masked_dataset=masked_imaging_7x7, image_path=plot_path
        )

        visualizer.visualize_imaging()

        assert plot_path + "subplots/subplot_imaging.png" in plot_patch.paths
        assert plot_path + "imaging/image.png" in plot_patch.paths
        assert plot_path + "imaging/noise_map.png" not in plot_patch.paths
        assert plot_path + "imaging/psf.png" in plot_patch.paths
        assert plot_path + "imaging/signal_to_noise_map.png" not in plot_patch.paths
        assert (
            plot_path + "imaging/absolute_signal_to_noise_map.png"
            not in plot_patch.paths
        )
        assert plot_path + "imaging/potential_chi_squared_map.png" in plot_patch.paths

    def test__visualizes_fit_and_inversion_using_configs(
        self,
        masked_imaging_7x7,
        fit_imaging_7x7,
        gaussians,
        include_all,
        plot_path,
        plot_patch,
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.PhaseImagingVisualizer(
            masked_dataset=masked_imaging_7x7, image_path=plot_path
        )

        visualizer.visualize_fit(
            fit=fit_imaging_7x7, gaussians=gaussians, during_analysis=False
        )

        assert plot_path + "subplots/subplot_fit_imaging.png" in plot_patch.paths
        assert plot_path + "fit_imaging/image.png" in plot_patch.paths
        assert plot_path + "fit_imaging/noise_map.png" not in plot_patch.paths
        assert plot_path + "fit_imaging/signal_to_noise_map.png" not in plot_patch.paths
        assert plot_path + "fit_imaging/model_image.png" in plot_patch.paths
        assert plot_path + "fit_imaging/residual_map.png" not in plot_patch.paths
        assert plot_path + "fit_imaging/normalized_residual_map.png" in plot_patch.paths
        assert plot_path + "fit_imaging/chi_squared_map.png" in plot_patch.paths

        image = aa.util.array.numpy_array_2d_from_fits(
            file_path=plot_path + "fit_imaging/fits/image.fits", hdu=0
        )

        assert image.shape == (5, 5)


class TestPhaseInterferometerVisualizer:
    def test__visualizes_interferometer_using_configs(
        self,
        masked_interferometer_7,
        general_config,
        include_all,
        plot_path,
        plot_patch,
    ):

        visualizer = vis.PhaseInterferometerVisualizer(
            masked_dataset=masked_interferometer_7, image_path=plot_path
        )

        visualizer.visualize_interferometer()

        assert plot_path + "subplots/subplot_interferometer.png" in plot_patch.paths
        assert plot_path + "interferometer/visibilities.png" in plot_patch.paths
        assert plot_path + "interferometer/u_wavelengths.png" not in plot_patch.paths
        assert plot_path + "interferometer/v_wavelengths.png" not in plot_patch.paths
        assert plot_path + "interferometer/primary_beam.png" in plot_patch.paths

    def test__visualizes_fit_using_configs(
        self,
        masked_interferometer_7,
        fit_interferometer_7,
        gaussians,
        include_all,
        plot_path,
        plot_patch,
    ):

        visualizer = vis.PhaseInterferometerVisualizer(
            masked_dataset=masked_interferometer_7, image_path=plot_path
        )

        visualizer.visualize_fit(
            fit=fit_interferometer_7, gaussians=gaussians, during_analysis=True
        )

        assert plot_path + "subplots/subplot_fit_interferometer.png" in plot_patch.paths
        assert plot_path + "fit_interferometer/visibilities.png" in plot_patch.paths
        assert plot_path + "fit_interferometer/noise_map.png" not in plot_patch.paths
        assert (
            plot_path + "fit_interferometer/signal_to_noise_map.png"
            not in plot_patch.paths
        )
        assert (
            plot_path + "fit_interferometer/model_visibilities.png" in plot_patch.paths
        )
        assert (
            plot_path + "fit_interferometer/residual_map_vs_uv_distances_real.png"
            not in plot_patch.paths
        )
        assert (
            plot_path
            + "fit_interferometer/normalized_residual_map_vs_uv_distances_real.png"
            in plot_patch.paths
        )
        assert (
            plot_path + "fit_interferometer/chi_squared_map_vs_uv_distances_real.png"
            in plot_patch.paths
        )
