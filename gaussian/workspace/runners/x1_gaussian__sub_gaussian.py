import os

# Welcome to the pipeline runner. This tool allows you to load dataset on strong lenses and pass it to pipelines for a
# strong lens analysis. To show you around, we'll load up some example dataset and run it through some of the example
# pipelines that come distributed with PyAutoLens, in the 'autolens_workspace/pipelines/simple' folder.

# In this runner, our analysis assumes there is a lens light component that requires modeling and subtraction.

# Why is this folder called 'simple'? Well, we actually recommend you break pipelines up. Details of how to do this
# can be found in the 'autolens_workspace/runners/advanced' folder, but its a bit conceptually difficult. So, to
# familiarize yourself with PyAutoLens, I'd stick to the simple pipelines for now!

# You should also checkout the 'autolens_workspace/runners/features' folder. Here, you'll find features that are
# available in pipelines, that allow you to customize the analysis.

# This runner is supplied as both this Python script and a Juypter notebook. Its up to you which you use - I personally
# prefer the python script as provided you keep it relatively small, its quick and easy to comment out different lens
# names and pipelines to set off different analyses. However, notebooks are a tidier way to manage visualization - so
# feel free to use notebooks. Or, use both for a bit, and decide your favourite!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import gaussian as g

# It is convenient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the function below.

dataset_label = "gaussian__sub_gaussian"
pixel_scales = 0.1

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_light_mass_and_x1_source/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label]
)

imaging = g.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = g.mask.unmasked(shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales)

g.plot.imaging.subplot_imaging(imaging=imaging)

### PIPELINE SETTINGS ###

# The'pipeline_settings' customize a pipeline's behaviour. Beginner pipelines only have one 'general' setting we'll
# change, which determines whether an external shear is fitted for in the mass model or not (default=True).

pipeline_general_settings = g.PipelineGeneralSettings()

from gaussian.workspace.pipelines.initialize import x1_gaussian

pipeline__initialize = x1_gaussian.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    phase_folders=["gaussian_x1__sub_gaussian", dataset_label],
)

from gaussian.workspace.pipelines.main import x1_gaussian__sub_gaussian

pipeline__main = x1_gaussian__sub_gaussian.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    phase_folders=["gaussian_x1__sub_gaussian", dataset_label],
)

pipeline = pipeline__initialize + pipeline__main

pipeline.run(dataset=imaging, mask=mask)

# Another example pipeline is shown below, which fits the dataset using a pixelized inversion for the source light.
# If you commented out the 3 lines of code above, and uncomment the code below, you can easily set this pipeline
# off running!

# from autolens_workspace.pipelines.simple import lens_sersic_sie_shear_source_inversion
# pipeline = lens_sersic_sie_shear_source_inversion.make_pipeline(phase_folders=[dataset_label, dataset_name])
# pipeline.run(dataset_label=imaging)
