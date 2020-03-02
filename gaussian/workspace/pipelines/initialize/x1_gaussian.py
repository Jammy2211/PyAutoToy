import autofit as af
import gaussian as g

# In this pipeline, we'll perform a basic analysis which fits a single Spherical Gaussian profile.

# Phase 1:

# Description: Fit the Spherical Gaussian profile.
# Profile: SphericalGaussian
# Previous Pipelines: None
# Prior Passing: None
# Notes: None


def make_pipeline(
    pipeline_general_settings,
    phase_folders=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    optimizer_class=af.MultiNest,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__initialize__x1_gaussian"

    # This pipeline's name is tagged according to whether:

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the Gaussian profile, where we:

    # 1) Set our priors on the Gaussian's (y,x) centre such that we assume the image is centred around the Gaussian.

    gaussian = af.PriorModel(g.SphericalGaussian)
    gaussian.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    gaussian.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = g.PhaseImaging(
        phase_name="phase_1__x1_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(gaussian_0=gaussian),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=optimizer_class,
    )

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout 'tutorial_7_multinest_black_magic' in
    # 'howtolens/chapter_2_lens_modeling'.

    # Fitting the lens galaxy and source galaxy from uninitialized priors often risks MultiNest getting stuck in a
    # local maxima, especially for the image in this example which actually has two source galaxies. Therefore, whilst
    # I will continue to use constant efficiency mode to ensure fast run time, I've upped the number of live points
    # and decreased the sampling efficiency from the usual values to ensure the non-linear search is robust.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.5

    return g.PipelineDataset(pipeline_name, phase1)
