import autofit as af
import numpy as np

from autoarray.structures import grids
from toy_gaussian.src import dimensions as dim
from toy_gaussian.src.model import geometry_profiles


class LightProfile(object):
    """Mixin class that implements functions common to all light profiles"""

    def profile_image_from_grid_radii(self, grid_radii):
        """
        Abstract method for obtaining intensity at on a grid of radii.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        raise NotImplementedError("intensity_at_radius should be overridden")

    # noinspection PyMethodMayBeStatic
    def profile_image_from_grid(self, grid, grid_radial_minimum=None):
        """
        Abstract method for obtaining intensity at a grid of Cartesian (y,x) coordinates.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        Returns
        -------
        intensity : ndarray
            The value of intensity at the given radius
        """
        raise NotImplementedError("profile_image_from_grid should be overridden")


# noinspection PyAbstractClass
class EllipticalLightProfile(geometry_profiles.EllipticalProfile, LightProfile):
    """Generic class for an elliptical light profiles"""

    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
    ):
        """  Abstract class for an elliptical light-profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalLightProfile, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        self.intensity = intensity

    @property
    def gaussian_centres(self):
        return [self.centre]


class EllipticalGaussian(EllipticalLightProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
        intensity: dim.Luminosity = 0.1,
        sigma: dim.Length = 0.01,
    ):
        """ The elliptical Gaussian light profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        intensity : float
            Overall intensity normalisation of the light profiles (electrons per second).
        sigma : float
            The sigma value of the Gaussian.
        """
        super(EllipticalGaussian, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi, intensity=intensity
        )
        self.sigma = sigma

    def profile_image_from_grid_radii(self, grid_radii):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii : float
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_radii, self.sigma))),
        )

    @grids.convert_coordinates_to_grid
    @geometry_profiles.transform_grid
    @geometry_profiles.move_grid_to_radial_minimum
    def profile_image_from_grid(self, grid, grid_radial_minimum=None):
        """
        Calculate the intensity of the light profile on a grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : ndarray
            The (y, x) coordinates in the original reference frame of the grid.
        """
        return self.profile_image_from_grid_radii(self.grid_to_elliptical_radii(grid))


class SphericalGaussian(EllipticalGaussian):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        intensity: dim.Luminosity = 0.1,
        sigma: dim.Length = 0.01,
    ):
        """ The spherical Gaussian light profile.

        Parameters
        ----------
        centre : (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        intensity : float_
            Overall intensity normalisation of the light profiles (electrons per second).
        sigma : float
            The sigma value of the Gaussian.
        """
        super(SphericalGaussian, self).__init__(
            centre=centre, axis_ratio=1.0, phi=0.0, intensity=intensity, sigma=sigma
        )
