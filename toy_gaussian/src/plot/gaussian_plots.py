from autoarray.plot import plotters
from toy_gaussian.src.plot import gaussian_plotters
from autoarray.util import plotter_util


@gaussian_plotters.set_include_and_plotter
@plotters.set_labels
def profile_image(gaussian, grid, positions=None, include=None, plotter=None):
    """Plot the image of a light profile, on a grid of (y,x) coordinates.

    Set *toy_gaussian.src.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    gaussian : model.profiles.gaussians.LightProfile
        The light profile whose image are plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.plot_array(
        array=gaussian.profile_image_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        gaussian_centres=include.gaussian_centres_from_gaussians(gaussians=[gaussian]),
        include_origin=include.origin,
    )


def luminosity_within_circle_in_electrons_per_second_as_function_of_radius(
    gaussian,
    minimum_radius=1.0e-4,
    maximum_radius=10.0,
    radii_bins=10,
    plot_axis_type="semilogy",
    plotter=None,
):

    radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        radii_points=radii_bins,
    )

    luminosities = list(
        map(
            lambda radius: gaussian.luminosity_within_circle_in_units(radius=radius),
            radii,
        )
    )

    plotter.plot_array(
        quantity=luminosities, radii=radii, plot_axis_type=plot_axis_type
    )
