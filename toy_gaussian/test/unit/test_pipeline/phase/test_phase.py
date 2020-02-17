import os
from os import path

import pytest

import autofit as af
import toy_gaussian as toy

from toy_gaussian.test.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    print("{}/../test_files/config/".format(directory))

    af.conf.instance = af.conf.Config("{}/../../test_files/config/".format(directory))


def clean_images():
    try:
        os.remove("{}/source_lens_phase/source_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/lens_image_0.fits".format(directory))
        os.remove("{}/source_lens_phase/model_image_0.fits".format(directory))
    except FileNotFoundError:
        pass
    af.conf.instance.dataset_path = directory


class TestPhase:
    def test__set_instances(self, phase_dataset_7x7):
        phase_dataset_7x7.gaussians = [toy.SphericalGaussian()]
        assert phase_dataset_7x7.model.gaussians == [toy.SphericalGaussian()]

    # def test__set_models(self, phase_dataset_7x7):
    #     phase_dataset_7x7.gaussians = af.PriorModel(cls=toy.SphericalGaussian)
    #     assert phase_dataset_7x7.model.gaussians[0] == type(toy.SphericalGaussian)

    def test__customize(
        self, mask_7x7, results_7x7, results_collection_7x7, imaging_7x7
    ):
        class MyPlanePhaseAnd(toy.PhaseImaging):
            def customize_priors(self, results):
                self.gaussians = results.last.instance.gaussians

        gaussian = toy.SphericalGaussian()
        gaussian_model = af.PriorModel(cls=toy.SphericalGaussian)

        setattr(results_7x7.instance, "gaussians", [gaussian])
        setattr(results_7x7.model, "gaussians", [gaussian_model])

        phase_dataset_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase", optimizer_class=mock_pipeline.MockNLO
        )

        phase_dataset_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results_collection_7x7
        )
        phase_dataset_7x7.customize_priors(results_collection_7x7)

        assert phase_dataset_7x7.gaussians == [gaussian]

        class MyPlanePhaseAnd(toy.PhaseImaging):
            def customize_priors(self, results):
                self.gaussians = results.last.model.gaussians

        gaussian = toy.SphericalGaussian()
        gaussian_model = af.PriorModel(cls=toy.SphericalGaussian)

        setattr(results_7x7.instance, "gaussians", [gaussian])
        setattr(results_7x7.model, "gaussians", [gaussian_model])

        phase_dataset_7x7 = MyPlanePhaseAnd(
            phase_name="test_phase", optimizer_class=mock_pipeline.MockNLO
        )

        phase_dataset_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results_collection_7x7
        )
        phase_dataset_7x7.customize_priors(results_collection_7x7)

        assert phase_dataset_7x7.gaussians == [gaussian_model]

    def test__duplication(self):
        phase_dataset_7x7 = toy.PhaseImaging(
            phase_name="test_phase",
            gaussians=[
                af.PriorModel(cls=toy.SphericalGaussian),
                af.PriorModel(cls=toy.SphericalGaussian),
            ],
        )

        toy.PhaseImaging(phase_name="test_phase")

        assert phase_dataset_7x7.gaussians is not None

    def test__phase_can_receive_list_of_gaussian_models(self):
        phase_dataset_7x7 = toy.PhaseImaging(
            gaussians=[
                af.PriorModel(cls=toy.SphericalGaussian),
                af.PriorModel(cls=toy.SphericalGaussian),
            ],
            optimizer_class=af.MultiNest,
            phase_name="test_phase",
        )

        for item in phase_dataset_7x7.model.path_priors_tuples:
            print(item)

        gaussian_0 = phase_dataset_7x7.model.gaussians[0]
        gaussian_1 = phase_dataset_7x7.model.gaussians[1]

        arguments = {
            gaussian_0.centre[0]: 0.1,
            gaussian_0.centre[1]: 0.2,
            gaussian_0.intensity.priors[0]: 0.3,
            gaussian_0.sigma.priors[0]: 0.4,
            gaussian_1.centre[0]: 0.5,
            gaussian_1.centre[1]: 0.6,
            gaussian_1.intensity.priors[0]: 0.7,
            gaussian_1.sigma.priors[0]: 0.8,
        }

        instance = phase_dataset_7x7.model.instance_for_arguments(arguments=arguments)

        assert instance.gaussians[0].centre[0] == 0.1
        assert instance.gaussians[0].centre[1] == 0.2
        assert instance.gaussians[0].intensity == 0.3
        assert instance.gaussians[0].sigma == 0.4
        assert instance.gaussians[1].centre[0] == 0.5
        assert instance.gaussians[1].centre[1] == 0.6
        assert instance.gaussians[1].intensity == 0.7
        assert instance.gaussians[1].sigma == 0.8


class TestResult:
    def test__results_of_phase_are_available_as_properties(self, imaging_7x7, mask_7x7):
        clean_images()

        phase_dataset_7x7 = toy.PhaseImaging(
            optimizer_class=mock_pipeline.MockNLO,
            gaussians=[toy.SphericalGaussian()],
            phase_name="test_phase_2",
        )

        result = phase_dataset_7x7.run(dataset=imaging_7x7, mask=mask_7x7)

        assert isinstance(result, toy.AbstractPhase.Result)
