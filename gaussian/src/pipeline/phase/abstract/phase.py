import autofit as af
from gaussian.src.pipeline.phase.abstract.result import Result


# noinspection PyAbstractClass


class AbstractPhase(af.AbstractPhase):
    Result = Result

    @af.convert_paths
    def __init__(self, paths, *, non_linear_class=af.MultiNest):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit
        models and hyper_gaussians passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        """

        super().__init__(paths=paths, non_linear_class=non_linear_class)

    @property
    def phase_folders(self):
        return self.optimizer.phase_folders

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is
            used in automated prior passing and should be overridden for any phase that
            contains its own PhasePropertys.
        """
        return []

    @property
    def path(self):
        return self.optimizer.path

    def make_result(self, result, analysis):
        return self.Result(
            instance=result.instance,
            log_likelihood=result.log_likelihood,
            previous_model=result.previous_model,
            gaussian_tuples=result.gaussian_tuples,
            analysis=analysis,
            optimizer=self.optimizer,
        )

    def run(self, dataset, mask, results=None):
        raise NotImplementedError()
