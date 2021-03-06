import autofit as af
from toy_gaussian.src.pipeline import phase_tagging
from toy_gaussian.src.pipeline.phase import dataset
from toy_gaussian.src.pipeline.phase.imaging.analysis import Analysis
from toy_gaussian.src.pipeline.phase.imaging.meta_imaging_fit import MetaImagingFit
from toy_gaussian.src.pipeline.phase.imaging.result import Result


class PhaseImaging(dataset.PhaseDataset):
    gaussians = af.PhaseProperty("gaussians")

    Analysis = Analysis
    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        gaussians=None,
        optimizer_class=af.MultiNest,
        sub_size=2,
        signal_to_noise_limit=None,
        bin_up_factor=None,
    ):

        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper_gaussians
        passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        sub_size: int
            The side length of the subgrid
        """

        phase_tag = phase_tagging.phase_tag_from_phase_settings(
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
        )
        paths.phase_tag = phase_tag

        super().__init__(paths, gaussians=gaussians, optimizer_class=optimizer_class)

        self.meta_imaging_fit = MetaImagingFit(
            model=self.model,
            bin_up_factor=bin_up_factor,
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
        )

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def modify_image(self, image, results):
        """
        Customize an masked_imaging. e.g. removing lens light.

        Parameters
        ----------
        image: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous lens

        Returns
        -------
        masked_imaging: scaled_array.ScaledSquarePixelArray
            The modified image (not changed by default)
        """
        return image

    def make_analysis(self, dataset, mask, results=None):
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
        self.meta_imaging_fit.model = self.model
        modified_image = self.modify_image(image=dataset.image, results=results)

        masked_imaging = self.meta_imaging_fit.masked_dataset_from(
            dataset=dataset, mask=mask, results=results, modified_image=modified_image
        )

        self.output_phase_info()

        analysis = self.Analysis(
            masked_imaging=masked_imaging,
            image_path=self.optimizer.paths.image_path,
            results=results,
        )

        return analysis

    def output_phase_info(self):

        file_phase_info = "{}/{}".format(
            self.optimizer.paths.phase_output_path, "phase.info"
        )

        with open(file_phase_info, "w") as phase_info:
            phase_info.write("Optimizer = {} \n".format(type(self.optimizer).__name__))
            phase_info.write(
                "Sub-grid size = {} \n".format(self.meta_imaging_fit.sub_size)
            )

            phase_info.close()
