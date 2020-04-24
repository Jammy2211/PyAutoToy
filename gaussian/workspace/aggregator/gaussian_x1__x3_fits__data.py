from pathlib import Path

import autofit as af
import autoarray.plot as aplt

# Before reading this tutorial, you should read the script '/aggregator/gaussian_x1__x3_fit_data.py'.

# Below, we set up the aggregator as we did in the previous tutorial.

workspace_path = Path(__file__).parent.parent
output_path = workspace_path / "output"

af.conf.instance = af.conf.Config(
    config_path=str(workspace_path / "config"), output_path=str(output_path)
)

aggregator = af.Aggregator(directory=str(output_path) + "/gaussian_x1__x3_fits")

# In the previous tutorial, we used the aggregator to load classes that provided an interface between the results of a
# non-linear search and this Python script. In this tutorial, we're going to use these interfaces to visualize results
# and fits of a non-linear search.

# We can use the aggregator to create a list of the data-sets fitted by a pipeline. The results in this list will be in
# the same order as the non-linear outputs, meaning we can easily use their results to fit these data-sets.
pipeline_name = "pipeline__main__x1_gaussian"
datasets = aggregator.filter(pipeline=pipeline_name).dataset
print("Dataset")
print(datasets, "\n")

# We can plot instances of the dataset object:
# [aplt.Imaging.subplot_imaging(imaging=dataset) for dataset in datasets]
