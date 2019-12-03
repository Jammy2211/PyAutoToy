import autofit as af
import autoarray as aa
from toy_gaussian.src.pipeline.phase import abstract
from toy_gaussian.src.pipeline.phase.dataset.result import Result


def default_mask_function(shape_2d, pixel_scales):
    return aa.mask.unmasked(shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1)


class PhaseDataset(abstract.AbstractPhase):
    gaussians = af.PhaseProperty("gaussians")

    Result = Result

    @af.convert_paths
    def __init__(self, paths, gaussians=None, optimizer_class=af.MultiNest):
        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper_gaussians
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        """

        super(PhaseDataset, self).__init__(paths, optimizer_class=optimizer_class)
        self.gaussians = gaussians or []

    def run(self, dataset, results=None, mask=None):
        """
        Run this phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        dataset: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_gaussians.
        """
        self.model = self.model.populate(results)

        analysis = self.make_analysis(dataset=dataset, results=results, mask=mask)

        self.customize_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, results=None, mask=None):
        """
        Create an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        mask: Mask
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """
        raise NotImplementedError()
