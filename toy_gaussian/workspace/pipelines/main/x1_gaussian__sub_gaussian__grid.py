import autofit as af
import toy_gaussian as toy


# In this pipeline, we'll perform a basic analysis which fits two Spherical Gaussian profiles, where we anticipate the
# second component will be a fainter and smaller Gaussian located within the main Gaussian, and only revealed after its
# subtraction.

# Phase 1:

# Description: Fit the Spherical Gaussian profile.
# Profile: SphericalGaussian
# Previous Pipelines: None
# Prior Passing: None
# Notes: None


def make_pipeline(
    phase_folders=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    parallel=False,
):
    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_main__x1_gaussian__sub_gaussian"

    pipeline_tag = toy.pipeline_tagging.pipeline_tag_from_pipeline_settings()

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the Gaussian profile, where we:

    # 1) Set our priors on the Gaussian's (y,x) centre such that we assume the image is centred around the Gaussian.

    class GridPhase(af.as_grid_search(phase_class=toy.PhaseImaging, parallel=parallel)):
        @property
        def grid_priors(self):
            return [
                self.model.gaussians.sub_gaussian.centre_0,
                self.model.gaussians.sub_gaussian.centre_1,
            ]

    sub_gaussian = af.PriorModel(cls=toy.SphericalGaussian)

    sub_gaussian.centre.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    sub_gaussian.centre.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    phase1 = GridPhase(
        phase_name="phase_1__x1_gaussian__sub_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(
            gaussian_0=af.last.instance.gaussians.gaussian_0, sub_gaussian=sub_gaussian
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
        number_of_steps=2,
    )

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout 'tutorial_7_multinest_black_magic' in
    # 'howtolens/chapter_2_lens_modeling'.

    # Fitting the lens galaxy and source galaxy from uninitialized priors often risks MultiNest getting stuck in a
    # local maxima, especially for the image in this example which actually has two source galaxies. Therefore, whilst
    # I will continue to use constant efficiency mode to ensure fast run time, I've upped the number of live points
    # and decreased the sampling efficiency from the usual values to ensure the non-linear search is robust.

    phase1.optimizer.const_efficiency_mode = False
    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.5

    phase2 = toy.PhaseImaging(
        phase_name="phase_2__subhalo_refine",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(
        gaussian_0=af.last[-1].model.gaussians.gaussian_0,
        sub_gaussian=phase1.result.model.gaussians.sub_gaussian,
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 80
    phase2.optimizer.sampling_efficiency = 0.3

    return toy.PipelineDataset(pipeline_name, phase1, phase2)
