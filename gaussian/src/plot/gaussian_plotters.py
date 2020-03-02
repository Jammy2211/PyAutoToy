import autofit as af
from autoarray.plot import plotters
from gaussian.src.plot import mat_objs
from autoarray.operators.inversion import mappers
import numpy as np
from functools import wraps
import copy


class GaussianPlotter(plotters.AbstractPlotter):
    def __init__(
        self,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        ticks=None,
        labels=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        liner=None,
        voronoi_drawer=None,
        gaussian_centres_scatterer=None,
    ):

        super(GaussianPlotter, self).__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            liner=liner,
            voronoi_drawer=voronoi_drawer,
        )

        if isinstance(self, Plotter):
            from_subplot_config = False
        else:
            from_subplot_config = True

        self.gaussian_centres_scatterer = (
            gaussian_centres_scatterer
            if gaussian_centres_scatterer is not None
            else mat_objs.GaussianCentreScatterer(
                from_subplot_config=from_subplot_config
            )
        )

    def plot_gaussian_attributes(self, gaussian_centres=None):

        if gaussian_centres is not None:
            self.gaussian_centres_scatterer.scatter_grids(grids=gaussian_centres)

    def plot_array(
        self,
        array,
        mask=None,
        positions=None,
        grid=None,
        lines=None,
        gaussian_centres=None,
        include_origin=False,
        include_border=False,
        bypass_output=False,
    ):
        """Plot an array of data_type as a figure.

        Parameters
        -----------
        settings : PlotterSettings
            Settings
        include : PlotterInclude
            Include
        labels : PlotterLabels
            labels
        outputs : PlotterOutputs
            outputs
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        origin : (float, float).
            The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
        mask : data_type.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        extract_array_from_mask : bool
            The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
            bright features outside the mask do not impact the color map of the plotters.
        zoom_around_mask : bool
            If True, the 2D region of the array corresponding to the rectangle encompassing all unmasked values is \
            plotted, thereby zooming into the region of interest.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
        positions : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data_type.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.
        as_subplot : bool
            Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        aspect : str
            The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
            the figure size ('auto').
        cmap : str
            The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
        norm : str
            The normalization of the colormap used to plotters the image, specifically whether it is linear ('linear'), log \
            ('log') or a symmetric log normalization ('symmetric_log').
        norm_min : float or None
            The minimum array value the colormap map spans (all values below this value are plotted the same color).
        norm_max : float or None
            The maximum array value the colormap map spans (all values above this value are plotted the same color).
        linthresh : float
            For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
            is linear.
        linscale : float
            For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
            relative to the logarithmic range.
        cb_ticksize : int
            The size of the tick labels on the colorbar.
        cb_fraction : float
            The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
        cb_pad : float
            Pads the color bar in the figure, which resizes the colorbar relative to the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        mask_scatterer : int
            The size of the points plotted to show the mask.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'

        Returns
        --------
        None

        Examples
        --------
            plotters.plot_array(
            array=image, origin=(0.0, 0.0), mask=circular_mask,
            border=False, points=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
            unit_label='scaled', kpc_per_arcsec=None, figsize=(7,7), aspect='auto',
            cmap='jet', norm='linear, norm_min=None, norm_max=None, linthresh=None, linscale=None,
            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
            title='Image', titlesize=16, xsize=16, ysize=16, xyticksize=16,
            mask_scatterer=10, border_pointsize=2, position_pointsize=10, grid_pointsize=10,
            xticks_manual=None, yticks_manual=None,
            output_path='/path/to/output', output_format='png', output_filename='image')
        """

        if array is None or np.all(array == 0):
            return

        array = array.in_1d_binned

        if array.mask.is_all_false:
            buffer = 0
        else:
            buffer = 1

        array = array.zoomed_around_mask(buffer=buffer)

        super(GaussianPlotter, self).plot_array(
            array=array,
            mask=mask,
            positions=positions,
            grid=grid,
            lines=lines,
            include_origin=include_origin,
            include_border=include_border,
            bypass_output=True,
        )

        self.plot_gaussian_attributes(gaussian_centres=gaussian_centres)

        if not bypass_output:
            self.output.to_figure(structure=array)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_grid(
        self,
        grid,
        color_array=None,
        axis_limits=None,
        indexes=None,
        positions=None,
        gaussian_centres=None,
        include_origin=False,
        include_border=False,
        symmetric_around_centre=True,
        bypass_output=False,
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotters of points.

        Parameters
        -----------
        grid : data_type.array.aa.Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        indexes : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        as_subplot : bool
            Whether the grid is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        label_yunits : str
            The label of the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        pointsize : int
            The size of the points plotted on the grid.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
        """

        super(GaussianPlotter, self).plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=axis_limits,
            indexes=indexes,
            positions=positions,
            symmetric_around_centre=symmetric_around_centre,
            include_origin=include_origin,
            include_border=include_border,
            bypass_output=True,
        )

        self.plot_gaussian_attributes(gaussian_centres=gaussian_centres)

        if not bypass_output:
            self.output.to_figure(structure=grid)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
        bypass_output=False,
    ):

        super(GaussianPlotter, self).plot_line(
            y=y,
            x=x,
            label=label,
            plot_axis_type=plot_axis_type,
            vertical_lines=vertical_lines,
            vertical_line_labels=vertical_line_labels,
            bypass_output=True,
        )

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        gaussian_centres=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self.plot_rectangular_mapper(
                mapper=mapper,
                source_pixel_values=source_pixel_values,
                positions=positions,
                gaussian_centres=gaussian_centres,
                include_origin=include_origin,
                include_grid=include_grid,
                include_pixelization_grid=include_pixelization_grid,
                include_border=include_border,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
            )

        else:

            self.plot_voronoi_mapper(
                mapper=mapper,
                source_pixel_values=source_pixel_values,
                positions=positions,
                gaussian_centres=gaussian_centres,
                include_origin=include_origin,
                include_grid=include_grid,
                include_pixelization_grid=include_pixelization_grid,
                include_border=include_border,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
            )

    def plot_rectangular_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        gaussian_centres=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        super(GaussianPlotter, self).plot_rectangular_mapper(
            mapper=mapper,
            source_pixel_values=source_pixel_values,
            positions=positions,
            include_origin=include_origin,
            include_pixelization_grid=include_pixelization_grid,
            include_grid=include_grid,
            include_border=include_border,
            image_pixel_indexes=image_pixel_indexes,
            source_pixel_indexes=source_pixel_indexes,
            bypass_output=True,
        )

        self.plot_gaussian_attributes(gaussian_centres=gaussian_centres)

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_voronoi_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        gaussian_centres=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        super(GaussianPlotter, self).plot_voronoi_mapper(
            mapper=mapper,
            source_pixel_values=source_pixel_values,
            positions=positions,
            include_origin=include_origin,
            include_pixelization_grid=include_pixelization_grid,
            include_grid=include_grid,
            include_border=include_border,
            image_pixel_indexes=image_pixel_indexes,
            source_pixel_indexes=source_pixel_indexes,
            bypass_output=True,
        )

        self.plot_gaussian_attributes(gaussian_centres=gaussian_centres)

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()


class Plotter(GaussianPlotter, plotters.Plotter):
    def __init__(
        self,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        ticks=None,
        labels=None,
        legend=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        liner=None,
        voronoi_drawer=None,
        gaussian_centres_scatterer=None,
    ):

        super(Plotter, self).__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            liner=liner,
            voronoi_drawer=voronoi_drawer,
            gaussian_centres_scatterer=gaussian_centres_scatterer,
        )


class SubPlotter(GaussianPlotter, plotters.SubPlotter):
    def __init__(
        self,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        legend=None,
        ticks=None,
        labels=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        liner=None,
        voronoi_drawer=None,
        gaussian_centres_scatterer=None,
    ):

        super(SubPlotter, self).__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            liner=liner,
            voronoi_drawer=voronoi_drawer,
            gaussian_centres_scatterer=gaussian_centres_scatterer,
        )


class Include(plotters.Include):
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        positions=None,
        gaussian_centres=None,
        inversion_pixelization_grid=None,
        inversion_grid=None,
        inversion_border=None,
        inversion_image_pixelization_grid=None,
    ):

        super(Include, self).__init__(
            origin=origin,
            mask=mask,
            grid=grid,
            border=border,
            inversion_pixelization_grid=inversion_pixelization_grid,
            inversion_grid=inversion_grid,
            inversion_border=inversion_border,
            inversion_image_pixelization_grid=inversion_image_pixelization_grid,
        )

        self.positions = self.load_include(value=positions, name="positions")
        self.gaussian_centres = self.load_include(
            value=gaussian_centres, name="gaussian_centres"
        )

    @staticmethod
    def load_include(value, name):

        return (
            af.conf.instance.visualize_general.get(
                section_name="include", attribute_name=name, attribute_type=bool
            )
            if value is None
            else value
        )

    def positions_from_masked_dataset(self, masked_dataset):

        if self.positions:
            return masked_dataset.positions
        else:
            return None

    def gaussian_centres_from_gaussians(self, gaussians):

        if self.gaussian_centres:
            return [gaussian.centre for gaussian in gaussians]
        else:
            return None

    def positions_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        mask : bool
            If *True*, the masks is plotted on the fit's datas.
        """
        if self.positions:
            return fit.masked_dataset.positions
        else:
            return None

    def inversion_image_pixelization_grid_from_fit(self, fit):

        if fit.inversion is not None:
            if self.inversion_image_pixelization_grid:
                if fit.inversion.mapper.is_image_plane_pixelization:
                    return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                        grid=fit.grid
                    )[-1]

        return None


def set_include_and_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = plotters.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = plotters.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = Plotter()
            plotter_key = "plotter"

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_include_and_sub_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = plotters.include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = plotters.plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = SubPlotter()
            plotter_key = "sub_plotter"

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def plot_array(
    array,
    mask=None,
    positions=None,
    grid=None,
    gaussian_centres=None,
    include=None,
    plotter=None,
):

    if include is None:
        include = Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_array(
        array=array,
        mask=mask,
        gaussian_centres=gaussian_centres,
        positions=positions,
        grid=grid,
        include_origin=include.origin,
        include_border=include.border,
    )


def plot_grid(
    grid,
    color_array=None,
    axis_limits=None,
    indexes=None,
    positions=None,
    gaussian_centres=None,
    symmetric_around_centre=True,
    include=None,
    plotter=None,
):

    if include is None:
        include = Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_grid(
        grid=grid,
        color_array=color_array,
        axis_limits=axis_limits,
        indexes=indexes,
        positions=positions,
        gaussian_centres=gaussian_centres,
        symmetric_around_centre=symmetric_around_centre,
        include_origin=include.origin,
        include_border=include.border,
    )


def plot_line(
    y,
    x,
    label=None,
    plot_axis_type="semilogy",
    vertical_lines=None,
    vertical_line_labels=None,
    plotter=None,
):

    if plotter is None:
        plotter = Plotter()

    plotter.plot_line(
        y=y,
        x=x,
        label=label,
        plot_axis_type=plot_axis_type,
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
    )


def plot_mapper_obj(
    mapper,
    gaussian_centres=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    if include is None:
        include = Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_mapper(
        mapper=mapper,
        include_grid=include.inversion_grid,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_border=include.inversion_border,
        gaussian_centres=gaussian_centres,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
    )
