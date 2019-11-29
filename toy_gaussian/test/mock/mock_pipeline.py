import autofit as af

class MockAnalysis(object):
    def __init__(self, value):
        self.value = value

    def fit(self, instance):
        return 1

class MockResults(object):
    def __init__(
        self, model_image=None, mask=None, instance=None, analysis=None, optimizer=None
    ):
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.mask = mask
        self.instance = instance or af.ModelInstance()
        self.model = af.ModelMapper()
        self.analysis = analysis
        self.optimizer = optimizer


class MockResult:
    def __init__(self, instance, figure_of_merit, model=None):
        self.instance = instance
        self.figure_of_merit = figure_of_merit
        self.model = model
        self.previous_model = model
        self.gaussian_tuples = None
        self.mask_2d = None


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis, model):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)

                likelihood = analysis.fit(instance)
                self.result = MockResult(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(model.instance_from_physical_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result
