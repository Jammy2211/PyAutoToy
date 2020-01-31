from __future__ import division, print_function

import pytest

import autofit as af
import autoarray as aa
import toy_gaussian as toy


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    af.conf.instance = af.conf.default


grid = aa.grid_irregular.manual_1d([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestGaussian:
    def test__constructor_and_units(self):
        gaussian = toy.EllipticalGaussian(
            centre=(1.0, 2.0), axis_ratio=0.5, phi=45.0, intensity=1.0, sigma=0.1
        )

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], toy.dim.Length)
        assert isinstance(gaussian.centre[1], toy.dim.Length)
        assert gaussian.centre[0].unit == "arcsec"
        assert gaussian.centre[1].unit == "arcsec"

        assert gaussian.axis_ratio == 0.5
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == 45.0
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, toy.dim.Luminosity)
        assert gaussian.intensity.unit == "eps"

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, toy.dim.Length)
        assert gaussian.sigma.unit_length == "arcsec"

        gaussian = toy.SphericalGaussian(centre=(1.0, 2.0), intensity=1.0, sigma=0.1)

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], toy.dim.Length)
        assert isinstance(gaussian.centre[1], toy.dim.Length)
        assert gaussian.centre[0].unit == "arcsec"
        assert gaussian.centre[1].unit == "arcsec"

        assert gaussian.axis_ratio == 1.0
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == 0.0
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, toy.dim.Luminosity)
        assert gaussian.intensity.unit == "eps"

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, toy.dim.Length)
        assert gaussian.sigma.unit_length == "arcsec"

    def test__intensity_as_radius__correct_value(self):
        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=1.0) == pytest.approx(
            0.24197, 1e-2
        )

        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=1.0) == pytest.approx(
            2.0 * 0.24197, 1e-2
        )

        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=1.0) == pytest.approx(
            0.1760, 1e-2
        )

        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )
        assert gaussian.profile_image_from_grid_radii(grid_radii=3.0) == pytest.approx(
            0.0647, 1e-2
        )

    def test__intensity_from_grid__same_values_as_above(self):
        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(0.24197, 1e-2)

        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0, sigma=1.0
        )

        assert gaussian.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )

        assert gaussian.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(0.1760, 1e-2)

        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )

        assert gaussian.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[0.0, 3.0]])
        ) == pytest.approx(0.0647, 1e-2)

        value = gaussian.profile_image_from_grid(
            grid=aa.coordinates(coordinates=[[(0.0, 3.0)]])
        )

        assert value[0][0] == pytest.approx(0.0647, 1e-2)

    def test__intensity_from_grid__change_geometry(self):
        gaussian = toy.EllipticalGaussian(
            centre=(1.0, 1.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[1.0, 0.0]])
        ) == pytest.approx(0.24197, 1e-2)

        gaussian = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[1.0, 0.0]])
        ) == pytest.approx(0.05399, 1e-2)

        gaussian_0 = toy.EllipticalGaussian(
            centre=(-3.0, -0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )

        gaussian_1 = toy.EllipticalGaussian(
            centre=(3.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )

        assert gaussian_0.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        ) == pytest.approx(
            gaussian_1.profile_image_from_grid(
                grid=aa.grid_irregular.manual_1d([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
            ),
            1e-4,
        )

        gaussian_0 = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=0.5, phi=180.0, intensity=1.0, sigma=1.0
        )

        gaussian_1 = toy.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=0.5, phi=0.0, intensity=1.0, sigma=1.0
        )

        assert gaussian_0.profile_image_from_grid(
            grid=aa.grid_irregular.manual_1d([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        ) == pytest.approx(
            gaussian_1.profile_image_from_grid(
                grid=aa.grid_irregular.manual_1d([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
            ),
            1e-4,
        )

    def test__spherical_and_elliptical_match(self):
        elliptical = toy.EllipticalGaussian(
            axis_ratio=1.0, phi=0.0, intensity=3.0, sigma=2.0
        )
        spherical = toy.SphericalGaussian(intensity=3.0, sigma=2.0)

        assert (
            elliptical.profile_image_from_grid(grid=grid)
            == spherical.profile_image_from_grid(grid=grid)
        ).all()

    def test__output_image_is_autoarray(self):
        grid = aa.grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        gaussian = toy.EllipticalGaussian()

        image = gaussian.profile_image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)

        gaussian = toy.SphericalGaussian()

        image = gaussian.profile_image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)
