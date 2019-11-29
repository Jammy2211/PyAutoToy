import os

import autofit as af
from toy_gaussian.src.plotters import phase_plotters


def setting(section, name):
    return af.conf.instance.visualize.get(section, name, bool)


def plot_setting(name):
    return setting("plots", name)


def figure_setting(name):
    return setting("figures", name)


class AbstractVisualizer:
    def __init__(self, image_path):
        self.image_path = image_path or ""
        try:
            os.makedirs(self.image_path)
        except (FileExistsError, FileNotFoundError):
            pass
        self.unit_label = af.conf.instance.visualize.get(
            "figures", "unit_label", str
        ).strip()
        self.include_mask = figure_setting("include_mask")


class SubPlotVisualizer(AbstractVisualizer):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.subplot_path = f"{image_path}subplots/"
        try:
            os.makedirs(self.subplot_path)
        except FileExistsError:
            pass


class PhaseDatasetVisualize(SubPlotVisualizer):
    def __init__(self, masked_dataset, image_path):
        super().__init__(image_path)
        self.masked_dataset = masked_dataset

        self.plot_dataset_as_subplot = plot_setting("plot_dataset_as_subplot")
        self.plot_dataset_data = plot_setting("plot_dataset_data")
        self.plot_dataset_noise_map = plot_setting("plot_dataset_noise_map")
        self.plot_dataset_psf = plot_setting("plot_dataset_psf")

        self.plot_dataset_signal_to_noise_map = plot_setting(
            "plot_dataset_signal_to_noise_map"
        )
        self.plot_dataset_absolute_signal_to_noise_map = plot_setting(
            "plot_dataset_absolute_signal_to_noise_map"
        )
        self.plot_dataset_potential_chi_squared_map = plot_setting(
            "plot_dataset_potential_chi_squared_map"
        )
        self.plot_fit_all_at_end_png = plot_setting("plot_fit_all_at_end_png")
        self.plot_fit_all_at_end_fits = plot_setting("plot_fit_all_at_end_fits")
        self.plot_fit_as_subplot = plot_setting("plot_fit_as_subplot")
        self.plot_fit_inversion_as_subplot = plot_setting(
            "plot_fit_inversion_as_subplot"
        )
        self.plot_fit_data = plot_setting("plot_fit_data")
        self.plot_fit_noise_map = plot_setting("plot_fit_noise_map")
        self.plot_fit_signal_to_noise_map = plot_setting("plot_fit_signal_to_noise_map")
        self.plot_fit_model_data = plot_setting("plot_fit_model_data")
        self.plot_fit_residual_map = plot_setting("plot_fit_residual_map")
        self.plot_fit_normalized_residual_map = plot_setting(
            "plot_fit_normalized_residual_map"
        )
        self.plot_fit_chi_squared_map = plot_setting("plot_fit_chi_squared_map")
        self.plot_fit_contribution_maps = plot_setting("plot_fit_contribution_maps")
        self.plot_fit_inversion_residual_map = plot_setting(
            "plot_fit_inversion_residual_map"
        )
        self.plot_fit_pixelization_normalized_residuals = plot_setting(
            "plot_fit_inversion_normalized_residual_map"
        )
        self.plot_fit_inversion_chi_squared_map = plot_setting(
            "plot_fit_inversion_chi_squared_map"
        )
        self.plot_fit_inversion_regularization_weights = plot_setting(
            "plot_fit_inversion_regularization_weight_map"
        )


class PhaseImagingVisualizer(PhaseDatasetVisualize):
    def __init__(self, masked_dataset, image_path):
        super(PhaseImagingVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_psf = plot_setting("plot_dataset_psf")

        self.plot_imaging()

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def plot_imaging(self):
        mask = self.masked_dataset.mask if self.include_mask else None

        phase_plotters.imaging_of_phase(
            imaging=self.masked_dataset.imaging,
            mask=mask,
            unit_label=self.unit_label,
            unit_conversion_factor=1.0,
            plot_as_subplot=self.plot_dataset_as_subplot,
            plot_image=self.plot_dataset_data,
            plot_noise_map=self.plot_dataset_noise_map,
            plot_psf=self.plot_dataset_psf,
            plot_signal_to_noise_map=self.plot_dataset_signal_to_noise_map,
            plot_absolute_signal_to_noise_map=self.plot_dataset_absolute_signal_to_noise_map,
            plot_potential_chi_squared_map=self.plot_dataset_potential_chi_squared_map,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_fit(self, fit, during_analysis):
        phase_plotters.imaging_fit_of_phase(
            fit=fit,
            during_analysis=during_analysis,
            include_mask=self.include_mask,
            plot_all_at_end_png=self.plot_fit_all_at_end_png,
            plot_all_at_end_fits=self.plot_fit_all_at_end_fits,
            plot_fit_as_subplot=self.plot_fit_as_subplot,
            plot_inversion_as_subplot=self.plot_fit_inversion_as_subplot,
            plot_image=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_image=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_inversion_residual_map=self.plot_fit_inversion_residual_map,
            plot_inversion_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_inversion_chi_squared_map=self.plot_fit_inversion_chi_squared_map,
            plot_inversion_regularization_weights=(
                self.plot_fit_inversion_regularization_weights
            ),
            unit_label=self.unit_label,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )
