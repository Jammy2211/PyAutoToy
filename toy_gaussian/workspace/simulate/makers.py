import autoarray as aa
import autofit as af
import toy_gaussian as toy

import os


def simulate_imaging_from_gaussian_and_output_to_fits(
    gaussians,
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
    image = sum(
        list(
            map(lambda gaussian: gaussian.profile_image_from_grid(grid=grid), gaussians)
        )
    )

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
        psf_path=dataset_path + "psf.fits",
        noise_map_path=dataset_path + "noise_map.fits",
        overwrite=True,
    )

    aa.plot.imaging.subplot(
        imaging=imaging,
        output_filename="imaging",
        output_path=dataset_path,
        format="png",
    )

    aa.plot.imaging.individual(
        imaging=imaging,
        plot_image=True,
        plot_noise_map=True,
        plot_signal_to_noise_map=True,
        output_path=dataset_path,
        format="png",
    )


def make__gaussian_x1(sub_size):

    data_type = "gaussian_x1"

    # This lens-only system has a Dev Vaucouleurs spheroid / bulge.

    gaussian = toy.SphericalGaussian(centre=(0.0, 0.0), intensity=10.0, sigma=0.5)

    simulate_imaging_from_gaussian_and_output_to_fits(
        gaussians=[gaussian],
        pixel_scales=0.1,
        shape_2d=(25, 25),
        data_type=data_type,
        sub_size=sub_size,
    )


def make__gaussian_x1__input_sigma(sub_size, sigma):

    data_type = "gaussian_x1__sigma_" + str(sigma)

    # This lens-only system has a Dev Vaucouleurs spheroid / bulge.

    gaussian = toy.SphericalGaussian(centre=(0.0, 0.0), intensity=10.0, sigma=sigma)

    simulate_imaging_from_gaussian_and_output_to_fits(
        gaussians=[gaussian],
        pixel_scales=0.1,
        shape_2d=(25, 25),
        data_type=data_type,
        sub_size=sub_size,
    )


def make__gaussian_x2(sub_size):

    data_type = "gaussian_x2"

    # This lens-only system has a Dev Vaucouleurs spheroid / bulge.

    gaussian_0 = toy.SphericalGaussian(centre=(0.0, 1.0), intensity=10.0, sigma=0.5)
    gaussian_1 = toy.SphericalGaussian(centre=(0.0, -1.0), intensity=10.0, sigma=0.5)

    simulate_imaging_from_gaussian_and_output_to_fits(
        gaussians=[gaussian_0, gaussian_1],
        pixel_scales=0.1,
        shape_2d=(40, 40),
        data_type=data_type,
        sub_size=sub_size,
    )


def make__gaussian__sub_gaussian(sub_size):

    data_type = "gaussian__sub_gaussian"

    # This lens-only system has a Dev Vaucouleurs spheroid / bulge.

    gaussian_0 = toy.SphericalGaussian(centre=(0.0, 0.0), intensity=10.0, sigma=1.0)
    gaussian_1 = toy.SphericalGaussian(centre=(0.5, 0.5), intensity=0.2, sigma=0.1)

    simulate_imaging_from_gaussian_and_output_to_fits(
        gaussians=[gaussian_0, gaussian_1],
        pixel_scales=0.1,
        shape_2d=(40, 40),
        data_type=data_type,
        sub_size=sub_size,
    )
