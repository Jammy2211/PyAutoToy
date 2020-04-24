import autofit as af
from autoarray.exc import InversionException
from autofit.exc import FitException
from autoarray.fit.fit import FitImaging
from gaussian.src.pipeline import visualizer


class Analysis(af.Analysis):
    def __init__(self, masked_imaging, image_path=None, results=None):

        self.visualizer = visualizer.PhaseImagingVisualizer(
            masked_dataset=masked_imaging, image_path=image_path
        )

        self.masked_imaging = masked_imaging

    def fit(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the masked_imaging in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model masked_imaging itself
        """

        try:
            fit = self.masked_imaging_fit_from_instance(instance=instance)
            return fit.log_likelihood
        except InversionException as e:
            raise FitException from e

    def masked_imaging_fit_from_instance(self, instance):

        gaussian_image = sum(
            list(
                map(
                    lambda gaussian: gaussian.profile_image_from_grid(
                        self.masked_imaging.grid
                    ),
                    instance.gaussians,
                )
            )
        ).in_1d_binned

        return FitImaging(
            masked_imaging=self.masked_imaging, model_image=gaussian_image
        )

    def visualize(self, instance, during_analysis):

        fit = self.masked_imaging_fit_from_instance(instance=instance)
        self.visualizer.visualize_fit(
            fit=fit, gaussians=instance.gaussians, during_analysis=during_analysis
        )
