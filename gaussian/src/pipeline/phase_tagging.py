def phase_tag_from_phase_settings(
    sub_size,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    real_space_shape_2d=None,
    real_space_pixel_scales=None,
):

    sub_size_tag = sub_size_tag_from_sub_size(sub_size=sub_size)
    signal_to_noise_limit_tag = signal_to_noise_limit_tag_from_signal_to_noise_limit(
        signal_to_noise_limit=signal_to_noise_limit
    )
    bin_up_factor_tag = bin_up_factor_tag_from_bin_up_factor(
        bin_up_factor=bin_up_factor
    )
    real_space_shape_2d_tag = real_space_shape_2d_tag_from_real_space_shape_2d(
        real_space_shape_2d=real_space_shape_2d
    )
    real_space_pixel_scales_tag = real_space_pixel_scales_tag_from_real_space_pixel_scales(
        real_space_pixel_scales=real_space_pixel_scales
    )

    return (
        "phase_tag"
        + real_space_shape_2d_tag
        + real_space_pixel_scales_tag
        + sub_size_tag
        + signal_to_noise_limit_tag
        + bin_up_factor_tag
    )


def sub_size_tag_from_sub_size(sub_size):
    """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

    This changes the phase name 'phase_name' as follows:

    sub_size = None -> phase_name
    sub_size = 1 -> phase_name_sub_size_2
    sub_size = 4 -> phase_name_sub_size_4
    """
    return "__sub_" + str(sub_size)


def signal_to_noise_limit_tag_from_signal_to_noise_limit(signal_to_noise_limit):
    """Generate a signal to noise limit tag, to customize phase names based on limiting the signal to noise ratio of
    the dataset being fitted.

    This changes the phase name 'phase_name' as follows:

    signal_to_noise_limit = None -> phase_name
    signal_to_noise_limit = 2 -> phase_name_snr_2
    signal_to_noise_limit = 10 -> phase_name_snr_10
    """
    if signal_to_noise_limit is None:
        return ""
    else:
        return "__snr_" + str(signal_to_noise_limit)


def bin_up_factor_tag_from_bin_up_factor(bin_up_factor):
    """Generate a bin up tag, to customize phase names based on the resolutioon the image is binned up by for faster \
    run times.

    This changes the phase name 'phase_name' as follows:

    bin_up_factor = 1 -> phase_name
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    bin_up_factor = 2 -> phase_name_bin_up_factor_2
    """
    if bin_up_factor == 1 or bin_up_factor is None:
        return ""
    else:
        return "__bin_" + str(bin_up_factor)


def real_space_shape_2d_tag_from_real_space_shape_2d(real_space_shape_2d):
    """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

    This changes the phase name 'phase_name' as follows:

    real_space_shape_2d = None -> phase_name
    real_space_shape_2d = 1 -> phase_name_real_space_shape_2d_2
    real_space_shape_2d = 4 -> phase_name_real_space_shape_2d_4
    """
    if real_space_shape_2d is None:
        return ""
    y = str(real_space_shape_2d[0])
    x = str(real_space_shape_2d[1])
    return "__rs_shape_" + y + "x" + x


def real_space_pixel_scales_tag_from_real_space_pixel_scales(real_space_pixel_scales):
    """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

    This changes the phase name 'phase_name' as follows:

    real_space_pixel_scales = None -> phase_name
    real_space_pixel_scales = 1 -> phase_name_real_space_pixel_scales_2
    real_space_pixel_scales = 4 -> phase_name_real_space_pixel_scales_4
    """
    if real_space_pixel_scales is None:
        return ""
    y = "{0:.2f}".format(real_space_pixel_scales[0])
    x = "{0:.2f}".format(real_space_pixel_scales[1])
    return "__rs_pix_" + y + "x" + x
