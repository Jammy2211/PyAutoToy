import autofit as af
from autofit.tools.phase import Dataset
from toy_gaussian.src.pipeline.phase import abstract
from toy_gaussian.src.pipeline.phase.dataset.result import Result


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

    def run(self, dataset: Dataset, mask, results=None):
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
        dataset.save(self.paths.phase_output_path)
        self.model = self.model.populate(results)

        analysis = self.make_analysis(dataset=dataset, mask=mask, results=results)

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
