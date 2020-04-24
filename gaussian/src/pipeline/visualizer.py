import autoarray as aa
import autofit as af
from autoarray.plot import mat_objs
from gaussian.src.plot import gaussian_plotters
from autoarray.plot import fit_imaging_plots, fit_interferometer_plots, inversion_plots


def setting(section, name):
    return af.conf.instance.visualize_plots.get(section, name, bool)


def plot_setting(section, name):
    return setting(section, name)


class AbstractVisualizer:
    def __init__(self, image_path):

        self.plotter = gaussian_plotters.Plotter(
            output=mat_objs.Output(path=image_path, format="png")
        )
        self.sub_plotter = gaussian_plotters.SubPlotter(
            output=mat_objs.Output(path=image_path + "subplots/", format="png")
        )
        self.include = gaussian_plotters.Include()


class PhaseDatasetVisualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, image_path):
        super().__init__(image_path)
        self.masked_dataset = masked_dataset

        self.plot_subplot_dataset = plot_setting("dataset", "subplot_dataset")
        self.plot_dataset_data = plot_setting("dataset", "data")
        self.plot_dataset_noise_map = plot_setting("dataset", "noise_map")
        self.plot_dataset_psf = plot_setting("dataset", "psf")

        self.plot_dataset_signal_to_noise_map = plot_setting(
            "dataset", "signal_to_noise_map"
        )
        self.plot_dataset_absolute_signal_to_noise_map = plot_setting(
            "dataset", "absolute_signal_to_noise_map"
        )
        self.plot_dataset_potential_chi_squared_map = plot_setting(
            "dataset", "potential_chi_squared_map"
        )

        self.plot_fit_all_at_end_png = plot_setting("fit", "all_at_end_png")
        self.plot_fit_all_at_end_fits = plot_setting("fit", "all_at_end_fits")
        self.plot_subplot_fit = plot_setting("fit", "subplot_fit")

        self.plot_fit_data = plot_setting("fit", "data")
        self.plot_fit_noise_map = plot_setting("fit", "noise_map")
        self.plot_fit_signal_to_noise_map = plot_setting("fit", "signal_to_noise_map")
        self.plot_fit_model_data = plot_setting("fit", "model_data")
        self.plot_fit_residual_map = plot_setting("fit", "residual_map")
        self.plot_fit_normalized_residual_map = plot_setting(
            "fit", "normalized_residual_map"
        )
        self.plot_fit_chi_squared_map = plot_setting("fit", "chi_squared_map")

        self.plot_subplot_inversion = plot_setting("inversion", "subplot_inversion")
        self.plot_inversion_reconstructed_image = plot_setting(
            "inversion", "reconstructed_image"
        )

        self.plot_inversion_reconstruction = plot_setting("inversion", "reconstruction")

        self.plot_inversion_errors = plot_setting("inversion", "errors")

        self.plot_inversion_residual_map = plot_setting("inversion", "residual_map")
        self.plot_inversion_normalized_residual_map = plot_setting(
            "inversion", "normalized_residual_map"
        )
        self.plot_inversion_chi_squared_map = plot_setting(
            "inversion", "chi_squared_map"
        )
        self.plot_inversion_regularization_weights = plot_setting(
            "inversion", "regularization_weight_map"
        )
        self.plot_inversion_interpolated_reconstruction = plot_setting(
            "inversion", "interpolated_reconstruction"
        )
        self.plot_inversion_interpolated_errors = plot_setting(
            "inversion", "interpolated_errors"
        )


class PhaseImagingVisualizer(PhaseDatasetVisualizer):
    def __init__(self, masked_dataset, image_path, results=None):
        super(PhaseImagingVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_psf = plot_setting("dataset", "psf")

        self.visualize_imaging()

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def visualize_imaging(self):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "imaging/"
        )

        if self.plot_subplot_dataset:
            aa.plot.Imaging.subplot_imaging(
                imaging=self.masked_imaging.imaging,
                mask=self.include.mask_from_masked_dataset(
                    masked_dataset=self.masked_dataset
                ),
                include=self.include,
                sub_plotter=self.sub_plotter,
            )

        aa.plot.Imaging.individual(
            imaging=self.masked_imaging.imaging,
            mask=self.include.mask_from_masked_dataset(
                masked_dataset=self.masked_dataset
            ),
            plot_image=self.plot_dataset_data,
            plot_noise_map=self.plot_dataset_noise_map,
            plot_psf=self.plot_dataset_psf,
            plot_signal_to_noise_map=self.plot_dataset_signal_to_noise_map,
            plot_absolute_signal_to_noise_map=self.plot_dataset_absolute_signal_to_noise_map,
            plot_potential_chi_squared_map=self.plot_dataset_potential_chi_squared_map,
            include=self.include,
            plotter=plotter,
        )

    def visualize_fit(self, fit, gaussians, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_imaging/"
        )

        if self.plot_subplot_fit:
            fit_imaging_plots.subplot_fit_imaging(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

        fit_imaging_plots.individuals(
            fit=fit,
            plot_image=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_image=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            include=self.include,
            plotter=plotter,
        )

        if fit.inversion is not None:

            if self.plot_subplot_inversion:
                inversion_plots.subplot_inversion(
                    inversion=fit.inversion,
                    image_positions=self.include.positions_from_fit(fit=fit),
                    grid=self.include.inversion_image_pixelization_grid_from_fit(
                        fit=fit
                    ),
                    gaussian_centres=self.include.gaussian_centres_from_gaussians(
                        gaussians=gaussians
                    ),
                    include=self.include,
                    sub_plotter=self.sub_plotter,
                )

            plotter = self.plotter.plotter_with_new_output(
                path=self.plotter.output.path + "inversion/"
            )

            inversion_plots.individuals(
                inversion=fit.inversion,
                image_positions=self.include.positions_from_fit(fit=fit),
                grid=self.include.inversion_image_pixelization_grid_from_fit(fit=fit),
                gaussian_centres=self.include.gaussian_centres_from_gaussians(
                    gaussians=gaussians
                ),
                plot_reconstructed_image=self.plot_inversion_reconstruction,
                plot_reconstruction=self.plot_inversion_reconstruction,
                plot_errors=self.plot_inversion_errors,
                plot_residual_map=self.plot_inversion_residual_map,
                plot_normalized_residual_map=self.plot_inversion_normalized_residual_map,
                plot_chi_squared_map=self.plot_inversion_chi_squared_map,
                plot_regularization_weight_map=self.plot_inversion_regularization_weights,
                plot_interpolated_reconstruction=self.plot_inversion_interpolated_reconstruction,
                plot_interpolated_errors=self.plot_inversion_interpolated_errors,
                include=self.include,
                plotter=plotter,
            )

        if not during_analysis:

            if self.plot_fit_all_at_end_png:
                fit_imaging_plots.individuals(
                    fit=fit,
                    plot_image=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_image=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    include=self.include,
                    plotter=plotter,
                )

                if fit.inversion is not None:
                    inversion_plots.individuals(
                        inversion=fit.inversion,
                        image_positions=self.include.positions_from_fit(fit=fit),
                        grid=self.include.inversion_image_pixelization_grid_from_fit(
                            fit=fit
                        ),
                        gaussian_centres=self.include.gaussian_centres_from_gaussians(
                            gaussians=gaussians
                        ),
                        plot_reconstructed_image=True,
                        plot_reconstruction=True,
                        plot_errors=True,
                        plot_residual_map=True,
                        plot_normalized_residual_map=True,
                        plot_chi_squared_map=True,
                        plot_regularization_weight_map=True,
                        plot_interpolated_reconstruction=True,
                        plot_interpolated_errors=True,
                        include=self.include,
                        plotter=plotter,
                    )

            if self.plot_fit_all_at_end_fits:

                self.visualize_fit_in_fits(fit=fit)

    def visualize_fit_in_fits(self, fit):

        fits_plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_imaging/fits/", format="fits"
        )

        fit_imaging_plots.individuals(
            fit=fit,
            plot_image=True,
            plot_noise_map=True,
            plot_signal_to_noise_map=True,
            plot_model_image=True,
            plot_residual_map=True,
            plot_normalized_residual_map=True,
            plot_chi_squared_map=True,
            include=self.include,
            plotter=fits_plotter,
        )

        if fit.inversion is not None:

            fits_plotter = self.plotter.plotter_with_new_output(
                path=self.plotter.output.path + "inversion/fits/", format="fits"
            )

            inversion_plots.individuals(
                inversion=fit.inversion,
                plot_reconstructed_image=True,
                plot_interpolated_reconstruction=True,
                plot_interpolated_errors=True,
                include=self.include,
                plotter=fits_plotter,
            )


class PhaseInterferometerVisualizer(PhaseDatasetVisualizer):
    def __init__(self, masked_dataset, image_path):
        super(PhaseInterferometerVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_uv_wavelengths = plot_setting("dataset", "uv_wavelengths")
        self.plot_dataset_primary_beam = plot_setting("dataset", "primary_beam")

        self.visualize_interferometer()

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    def visualize_interferometer(self):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "interferometer/"
        )

        if self.plot_subplot_dataset:
            aa.plot.Interferometer.subplot_interferometer(
                interferometer=self.masked_dataset.interferometer,
                include=self.include,
                sub_plotter=self.sub_plotter,
            )

        aa.plot.Interferometer.individual(
            interferometer=self.masked_dataset.interferometer,
            plot_visibilities=self.plot_dataset_data,
            plot_u_wavelengths=self.plot_dataset_uv_wavelengths,
            plot_v_wavelengths=self.plot_dataset_uv_wavelengths,
            plot_primary_beam=self.plot_dataset_primary_beam,
            include=self.include,
            plotter=plotter,
        )

    def visualize_fit(self, fit, gaussians, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_interferometer/"
        )

        if self.plot_subplot_fit:
            fit_interferometer_plots.subplot_fit_interferometer(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

        fit_interferometer_plots.individuals(
            fit=fit,
            plot_visibilities=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_visibilities=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            include=self.include,
            plotter=plotter,
        )

        if fit.inversion is not None:

            plotter = self.plotter.plotter_with_new_output(
                path=self.plotter.output.path + "inversion/"
            )

            # if self.plot_fit_inversion_as_subplot:
            #     inversion_plots.subplot_inversion(
            #         inversion=fit.inversion,
            #         image_positions=self.include.positions_from_fit(fit=fit),
            #         grid=self.include.inversion_image_pixelization_grid_from_fit(fit=fit),
            # gaussian_centres = self.include.gaussian_centres_from_gaussians(
            #     gaussians=gaussians
            # ),
            #         include=self.include,
            #         sub_plotter=self.sub_plotter,
            #     )

            inversion_plots.individuals(
                inversion=fit.inversion,
                image_positions=self.include.positions_from_fit(fit=fit),
                grid=self.include.inversion_image_pixelization_grid_from_fit(fit=fit),
                gaussian_centres=self.include.gaussian_centres_from_gaussians(
                    gaussians=gaussians
                ),
                plot_reconstructed_image=self.plot_inversion_reconstruction,
                plot_reconstruction=self.plot_inversion_reconstruction,
                plot_errors=self.plot_inversion_errors,
                #   plot_residual_map=self.plot_fit_inversion_residual_map,
                #   plot_normalized_residual_map=self.plot_fit_inversion_normalized_residual_map,
                #   plot_chi_squared_map=self.plot_fit_inversion_chi_squared_map,
                plot_regularization_weight_map=self.plot_inversion_regularization_weights,
                plot_interpolated_reconstruction=self.plot_inversion_interpolated_reconstruction,
                plot_interpolated_errors=self.plot_inversion_interpolated_errors,
                include=self.include,
                plotter=plotter,
            )

        if not during_analysis:

            if self.plot_fit_all_at_end_png:
                fit_interferometer_plots.individuals(
                    fit=fit,
                    plot_visibilities=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_visibilities=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    include=self.include,
                    plotter=plotter,
                )

                if fit.inversion is not None:
                    inversion_plots.individuals(
                        inversion=fit.inversion,
                        image_positions=self.include.positions_from_fit(fit=fit),
                        grid=self.include.inversion_image_pixelization_grid_from_fit(
                            fit=fit
                        ),
                        gaussian_centres=self.include.gaussian_centres_from_gaussians(
                            gaussians=gaussians
                        ),
                        plot_reconstructed_image=True,
                        plot_reconstruction=True,
                        plot_errors=True,
                        #     plot_residual_map=True,
                        #     plot_normalized_residual_map=True,
                        #     plot_chi_squared_map=True,
                        plot_regularization_weight_map=True,
                        plot_interpolated_reconstruction=True,
                        plot_interpolated_errors=True,
                        include=self.include,
                        plotter=plotter,
                    )

            if self.plot_fit_all_at_end_fits:
                fits_plotter = plotter.plotter_with_new_output(
                    path=plotter.output.path + "/fits/", format="fits"
                )

                fit_interferometer_plots.individuals(
                    fit=fit,
                    plot_visibilities=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_visibilities=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    include=self.include,
                    plotter=fits_plotter,
                )

                if fit.inversion is not None:
                    inversion_plots.individuals(
                        inversion=fit.inversion,
                        image_positions=self.include.positions_from_fit(fit=fit),
                        grid=self.include.inversion_image_pixelization_grid_from_fit(
                            fit=fit
                        ),
                        gaussian_centres=self.include.gaussian_centres_from_gaussians(
                            gaussians=gaussians
                        ),
                        plot_reconstructed_image=True,
                        plot_interpolated_reconstruction=True,
                        plot_interpolated_errors=True,
                        include=self.include,
                        plotter=plotter,
                    )
