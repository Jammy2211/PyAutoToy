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
    non_linear_class=af.MultiNest,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__initialize__x2_gaussian_separate"

    # This pipeline's name is tagged according to whether:

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the Gaussian profile, where we:

    # 1) Set our priors on the Gaussian's (y,x) centre such that we assume the Gaussian is centred around (0.0, -1.0).

    gaussian_0 = af.PriorModel(g.SphericalGaussian)
    gaussian_0.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    gaussian_0.centre_1 = af.GaussianPrior(mean=-1.0, sigma=0.1)

    phase1 = g.PhaseImaging(
        phase_name="phase_1__left_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(gaussian_0=gaussian_0),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.5

    ### PHASE 2 ###

    # In phase 2, we will fit the Gaussian profile, where we:

    # 1) Use the resulting Gaussian from the result of phase 1 to fit its light.
    # 2) Set our priors on the second Gaussian's (y,x) centre such that we assume the Gaussian is centred around (0.0, 1.0).

    gaussian_1 = af.PriorModel(g.SphericalGaussian)
    gaussian_1.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    gaussian_1.centre_1 = af.GaussianPrior(mean=1.0, sigma=0.1)

    phase2 = g.PhaseImaging(
        phase_name="phase_2__right_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(
            gaussian_0=phase1.result.instance.gaussians.gaussian_0,
            gaussian_1=gaussian_1,
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=non_linear_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.5

    ### PHASE 2 ###

    # In phase 3, we will fit both Gaussian profiles, where we:

    # 1) Use the resulting Gaussian models from the results of phase 1 and 2 to initialize their model parameters.

    phase3 = g.PhaseImaging(
        phase_name="phase_3__both_gaussian",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(
            gaussian_0=phase1.result.model.gaussians.gaussian_0,
            gaussian_1=phase2.result.model.gaussians.gaussian_1,
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=non_linear_class,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 20
    phase3.optimizer.sampling_efficiency = 0.5

    return g.PipelineDataset(pipeline_name, phase1, phase2, phase3)
