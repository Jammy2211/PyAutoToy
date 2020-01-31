import autofit as af
import autoarray as aa


def imaging_of_phase(
    imaging,
    mask,
    unit_conversion_factor,
    unit_label,
    plot_as_subplot,
    plot_image,
    plot_noise_map,
    plot_psf,
    plot_signal_to_noise_map,
    plot_absolute_signal_to_noise_map,
    plot_potential_chi_squared_map,
    visualize_path,
    subplot_path,
):

    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["imaging"]
    )

    if plot_as_subplot:

        aa.plot.imaging.subplot_imaging(
            imaging=imaging,
            mask=mask,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=subplot_path,
            format="png",
        )

    aa.plot.imaging.individual(
        imaging=imaging,
        mask=mask,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_psf=plot_psf,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_absolute_signal_to_noise_map=plot_absolute_signal_to_noise_map,
        plot_potential_chi_squared_map=plot_potential_chi_squared_map,
        output_path=output_path,
        format="png",
    )


def imaging_fit_of_phase(
    fit,
    during_analysis,
    mask,
    unit_label,
    unit_conversion_factor,
    plot_all_at_end_png,
    plot_all_at_end_fits,
    plot_fit_as_subplot,
    plot_inversion_as_subplot,
    plot_image,
    plot_noise_map,
    plot_signal_to_noise_map,
    plot_model_image,
    plot_residual_map,
    plot_normalized_residual_map,
    plot_chi_squared_map,
    plot_inversion_residual_map,
    plot_inversion_normalized_residual_map,
    plot_inversion_chi_squared_map,
    plot_inversion_regularization_weights,
    visualize_path,
    subplot_path,
):

    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["fit"]
    )

    if plot_fit_as_subplot:

        aa.plot.fit_imaging.subplot_fit_imaging(
            fit=fit,
            mask=mask,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            output_path=subplot_path,
            format="png",
        )

    aa.plot.fit_imaging.individuals(
        fit=fit,
        mask=mask,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        plot_image=plot_image,
        plot_noise_map=plot_noise_map,
        plot_signal_to_noise_map=plot_signal_to_noise_map,
        plot_model_image=plot_model_image,
        plot_residual_map=plot_residual_map,
        plot_chi_squared_map=plot_chi_squared_map,
        plot_normalized_residual_map=plot_normalized_residual_map,
        plot_inversion_residual_map=plot_inversion_residual_map,
        plot_inversion_normalized_residual_map=plot_inversion_normalized_residual_map,
        plot_inversion_chi_squared_map=plot_inversion_chi_squared_map,
        plot_inversion_regularization_weight_map=plot_inversion_regularization_weights,
        output_path=output_path,
        format="png",
    )

    if not during_analysis:

        if plot_all_at_end_png:

            aa.plot.fit_imaging.individuals(
                fit=fit,
                mask=mask,
                unit_label=unit_label,
                unit_conversion_factor=unit_conversion_factor,
                plot_image=True,
                plot_noise_map=True,
                plot_signal_to_noise_map=True,
                plot_model_image=True,
                plot_residual_map=True,
                plot_normalized_residual_map=True,
                plot_chi_squared_map=True,
                plot_inversion_residual_map=True,
                plot_inversion_normalized_residual_map=True,
                plot_inversion_chi_squared_map=True,
                plot_inversion_regularization_weight_map=True,
                output_path=output_path,
                format="png",
            )

        if plot_all_at_end_fits:

            fits_path = af.path_util.make_and_return_path_from_path_and_folder_names(
                path=output_path, folder_names=["fits"]
            )

            aa.plot.fit_imaging.individuals(
                fit=fit,
                mask=mask,
                unit_label=unit_label,
                unit_conversion_factor=unit_conversion_factor,
                plot_image=True,
                plot_noise_map=True,
                plot_signal_to_noise_map=True,
                plot_model_image=True,
                plot_residual_map=True,
                plot_normalized_residual_map=True,
                plot_chi_squared_map=True,
                plot_inversion_residual_map=True,
                plot_inversion_normalized_residual_map=True,
                plot_inversion_chi_squared_map=True,
                plot_inversion_regularization_weight_map=True,
                output_path=fits_path,
                output_format="fits",
            )
