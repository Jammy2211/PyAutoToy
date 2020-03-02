import os

import autofit as af
import gaussian as g


### AUTOFIT + CONFIG SETUP ###

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###


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

# g.plot.imaging.subplot(imaging=imaging)

### PIPELINE SETTINGS ###

# The'pipeline_settings' customize a pipeline's behaviour. Beginner pipelines only have one 'general' setting we'll
# change, which determines whether an external shear is fitted for in the mass model or not (default=True).

pipeline_general_settings = g.PipelineGeneralSettings()

from gaussian.workspace.pipelines.initialize import x1_gaussian

pipeline__initialize = x1_gaussian.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    phase_folders=["gaussian_x1__sub_gaussian__grid", dataset_label],
)

from gaussian.workspace.pipelines.main import x1_gaussian__sub_gaussian__grid

pipeline__main = x1_gaussian__sub_gaussian__grid.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    phase_folders=["gaussian_x1__sub_gaussian__grid", dataset_label],
    parallel=False,
)

pipeline = pipeline__initialize + pipeline__main

pipeline.run(dataset=imaging, mask=mask)
