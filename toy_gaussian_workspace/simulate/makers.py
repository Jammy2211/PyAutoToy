import autoarray as aa
import autofit as af
import toy_gaussian as toy

import os


def simulate_imaging_from_gaussian_and_output_to_fits(
    gaussian,
    pixel_scales,
    shape_2d,
    data_type,
    sub_size,
    exposure_time=300.0,
    background_level=1.0,
):

    # Setup the grid which will be used for generating the image of the Gaussian.
    grid = aa.grid.uniform(
        shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=sub_size
    )

    # Use this grid and our Gaussian profile to create an image of the Gaussian.
    image = gaussian.profile_image_from_grid(grid=grid)

    # Simulate the imaging data, including Poisson noise.
    imaging = aa.imaging.simulate(
        image=image.in_1d_binned,
        exposure_time=exposure_time,
        background_level=background_level,
        add_noise=True,
    )

    # Output this simulated imaging-data to the PyAutoToy/gaussian/dataset/spherical_gaussian folder.
    workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

    dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=workspace_path, folder_names=["dataset", data_type]
    )

    imaging.output_to_fits(
        image_path=dataset_path + "image.fits",
        noise_map_path=dataset_path + "noise_map.fits",
        overwrite=True,
    )

    aa.plot.imaging.subplot(
        imaging=imaging,
        output_filename="imaging",
        output_path=dataset_path,
        output_format="png",
    )

    aa.plot.imaging.individual(
        imaging=imaging,
        plot_image=True,
        plot_noise_map=True,
        plot_signal_to_noise_map=True,
        output_path=dataset_path,
        output_format="png",
    )


def make__gaussian(sub_size):

    data_type = "spherical_gaussian"

    # This lens-only system has a Dev Vaucouleurs spheroid / bulge.

    gaussian = toy.SphericalGaussian(centre=(0.0, 0.0), intensity=100.0, sigma=0.5)

    simulate_imaging_from_gaussian_and_output_to_fits(
        gaussian=gaussian,
        pixel_scales=0.1,
        shape_2d=(50, 50),
        data_type=data_type,
        sub_size=sub_size,
    )
