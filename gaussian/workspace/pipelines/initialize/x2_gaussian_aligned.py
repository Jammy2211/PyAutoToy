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

    pipeline_name = "pipeline__initialize__x2_gaussian_aligned"

    # This pipeline's name is tagged according to whether:

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the Gaussian profile, where we:

    # 1) Set our priors on the Gaussian's (y,x) centre such that we assume the Gaussian is centred around (0.0, -1.0).

    gaussian_0 = af.PriorModel(g.SphericalGaussian)
    gaussian_1 = af.PriorModel(g.SphericalGaussian)
    gaussian_0.centre = gaussian_1.centre

    gaussian_0.add_assertion(gaussian_0.sigma < gaussian_1.sigma)

    phase1 = g.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        gaussians=af.CollectionPriorModel(gaussian_0=gaussian_0, gaussian_1=gaussian_1),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.5

    return g.PipelineDataset(pipeline_name, phase1)
