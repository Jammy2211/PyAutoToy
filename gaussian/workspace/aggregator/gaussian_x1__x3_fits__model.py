from pathlib import Path

import autofit as af
import autoarray.plot as aplt

# After fitting a large suite of data with the same pipeline, the aggregator allows us to load the results and
# manipulate / plot them using a Python script or Jupyter notebook.

# In 'runners/gaussian_x1__x3_fits.py', we fitted 3 images of Gaussians which were simulated using different input
# sigma values of 0.1, 0.5 and 1.0. We fitted each image with the pipelines:

# - 'pipeline__initialize__gaussian_x1'
# - 'pipeline__main__gaussian_x1'.

# The results of this fit are in the 'output/gaussian_x1__x3_fits' folder.

# We can load the results of all 3 fits using the aggregator, enabling us to manipulate those results in this Python
# script or a Jupyter notebook to plot figures, interpret results, check specific values, etc.

# To begin, we setup the path to the gaussian_workspace and our output folder.
workspace_path = Path(__file__).parent.parent
output_path = workspace_path / "output"

# Now we'll use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=str(workspace_path / "config"), output_path=str(output_path)
)

# To use the aggregator we simply pass it the output folder of the results we want to load.
aggregator = af.Aggregator(directory=str(output_path) + "/gaussian_x1__x3_fits")

# We can now create a list of the 'non-linear outputs' of all fits, where an instance of the Output class acts as an
# interface between the results of the non-linear fit on your hard-disk and Python.
#
# The fits to each Gaussian used MultiNest, so this will create a list of instances of the MultiNestOutput class.
outputs = aggregator.output

# When we print this list of outputs, we will see 6 different MultiNestOutput instances. These corresponded to our 3
# fits, 3 of which are the initialization pipeline and 3 of which are the main pipeline.
print("MultiNest Outputs:")
print(outputs, "\n")

# The results of different pipelines generally arn't very useful. For instance, they may correspond to fits using
# different models, making it hard to extract all of the results from a fits.

# Lets get rid of the results of the initialization pipeline by passing the name of the main pipeline we want to
# load the results of to the aggregator's filter method.
pipeline_name = "pipeline__main__x1_gaussian"
outputs = aggregator.filter(pipeline=pipeline_name).output

# As expected, this list now has only 3 MultiNestOutputs.
print("Filtered MultiNest Outputs:")
print(outputs, "\n")

# We can, use these outputs to create a list of the most-likely (e.g. highest likelihood) model of each fit to our
# three images.
most_likely_vector = [out.most_probable_vector for out in outputs]
print("Most Likely Model Parameter Lists:")
print(most_likely_vector, "\n")

# This provides us with lists of all model parameters. However, this isn't that much use - which values correspond
# to which parameters?

# Its more use to create the model instance of every fit.
most_likely_instances = [out.most_probable_instance for out in outputs]
print("Most Likely Model Instances:")
print(most_likely_instances, "\n")

# A model instance uses the model defined by a pipeline. For our gaussians we can thus extract their parameters.
print("Most Likely Gaussian Sigmas:")
print([instance.gaussians.gaussian_0.sigma for instance in most_likely_instances])
print()

# We can also access the 'most probable' model, which is the model computed by marginalizing over the MultiNest samples
# of every parameter in 1D and taking the median of this PDF.
most_probable_vector = [out.most_probable_vector for out in outputs]
most_probable_instances = [out.most_probable_instance for out in outputs]

print("Most Probable Model Parameter Lists:")
print(most_probable_vector, "\n")
print("Most probable Model Instances:")
print(most_probable_instances, "\n")
print("Most Probable Gaussian Sigmas:")
print([instance.gaussians.gaussian_0.sigma for instance in most_probable_instances])
print()

# We can compute the upper and lower model errors at a given sigma limit.
upper_errors = [out.error_vector_at_upper_sigma(sigma=3.0) for out in outputs]
upper_error_instances = [
    out.error_instance_at_upper_sigma(sigma=3.0) for out in outputs
]
lower_errors = [out.error_vector_at_lower_sigma(sigma=3.0) for out in outputs]
lower_error_instances = [
    out.error_instance_at_lower_sigma(sigma=3.0) for out in outputs
]

print("Errors Lists:")
print(upper_errors, "\n")
print(lower_errors, "\n")
print("Errors Instances:")
print(upper_error_instances, "\n")
print(lower_error_instances, "\n")
print("Errors of Gaussian Sigmas:")
print([instance.gaussians.gaussian_0.sigma for instance in upper_error_instances])
print([instance.gaussians.gaussian_0.sigma for instance in lower_error_instances])
print()

# We can load the "model_results" of all phases, which is string that summarizes every fit's lens model providing
# quick inspection of all results.
results = aggregator.filter(pipeline=pipeline_name).model_results
print("Model Results Summary:")
print(results, "\n")

# We can also grab an instance of the dataset that was passed down the pipeline
datasets = aggregator.filter(pipeline=pipeline_name).dataset
print("Dataset")
print(datasets, "\n")

# We can plot instances of the dataset object:
[aplt.imaging.image(imaging=dataset) for dataset in datasets]
