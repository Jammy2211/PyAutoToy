import autoarray as aa
from os import path
import os
import pytest
import shutil

import numpy as np
import gaussian.src.plot as aplt


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return "{}/..//test_files/plot/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


class TestGaussianPlotterAttributes:
    def test__gaussian_centres_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.gaussian_centres_scatterer.size == 10
        assert plotter.gaussian_centres_scatterer.marker == "+"
        assert plotter.gaussian_centres_scatterer.colors == ["k", "r"]

        plotter = aplt.Plotter(
            gaussian_centres_scatterer=aplt.GaussianCentreScatterer(
                size=1, marker=".", colors="k"
            )
        )

        assert plotter.gaussian_centres_scatterer.size == 1
        assert plotter.gaussian_centres_scatterer.marker == "."
        assert plotter.gaussian_centres_scatterer.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.gaussian_centres_scatterer.size == 10
        assert sub_plotter.gaussian_centres_scatterer.marker == "+"
        assert sub_plotter.gaussian_centres_scatterer.colors == ["k", "r"]

        sub_plotter = aplt.SubPlotter(
            gaussian_centres_scatterer=aplt.GaussianCentreScatterer.sub(
                marker="o", colors="r"
            )
        )

        assert sub_plotter.gaussian_centres_scatterer.size == 10
        assert sub_plotter.gaussian_centres_scatterer.marker == "o"
        assert sub_plotter.gaussian_centres_scatterer.colors == ["r"]


class TestGaussianPlotterPlots:
    def test__plot_array__works_with_all_extras_included(self, plot_path, plot_patch):

        array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        mask = aa.Mask.circular(
            shape_2d=array.shape_2d,
            pixel_scales=array.pixel_scales,
            radius=5.0,
            centre=(2.0, 2.0),
        )

        grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array1", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=[(-1.0, -1.0)],
            gaussian_centres=[(1.0, 1.0)],
            include_origin=True,
            include_border=True,
        )

        assert plot_path + "array1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array2", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=[[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]],
            include_origin=True,
            include_border=True,
        )

        assert plot_path + "array2.png" in plot_patch.paths

        aplt.Array(
            array=array,
            mask=mask,
            grid=grid,
            positions=[(-1.0, -1.0)],
            gaussian_centres=[(1.0, 1.0)],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        assert plot_path + "array3.png" in plot_patch.paths

    def test__plot_array__fits_files_output_correctly(self, plot_path):

        plot_path = plot_path + "/fits/"

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        arr = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array", format="fits")
        )

        plotter.plot_array(array=arr)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=plot_path + "/array.fits", hdu=0
        )

        assert (arr == np.ones(shape=(31, 31))).all()

        mask = aa.Mask.circular(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), radius=5.0, centre=(2.0, 2.0)
        )

        masked_array = aa.MaskedArray.manual_2d(array=arr, mask=mask)

        plotter.plot_array(array=masked_array)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=plot_path + "/array.fits", hdu=0
        )

        assert arr.shape == (13, 13)

    def test__plot_grid__works_with_all_extras_included(self, plot_path, plot_patch):
        grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)
        color_array = np.linspace(start=0.0, stop=1.0, num=grid.shape_1d)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid1", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            gaussian_centres=[(1.0, 1.0)],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
        )

        assert plot_path + "grid1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid2", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            gaussian_centres=[(1.0, 1.0)],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert plot_path + "grid2.png" in plot_patch.paths

        aplt.Grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            gaussian_centres=[(1.0, 1.0)],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
        )

        assert plot_path + "grid3.png" in plot_patch.paths

    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line1", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plot_path + "line1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line2", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plot_path + "line2.png" in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            ),
        )

        assert plot_path + "line3.png" in plot_patch.paths

    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            gaussian_centres=[(1.0, 1.0)],
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            gaussian_centres=[(1.0, 1.0)],
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper2.png" in plot_patch.paths

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            gaussian_centres=[(1.0, 1.0)],
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert plot_path + "mapper3.png" in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper2.png" in plot_patch.paths

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert plot_path + "mapper3.png" in plot_patch.paths
