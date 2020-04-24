import autofit as af


class MockAnalysis:
    def __init__(self, value):
        self.value = value

    def fit(self, instance):
        return 1


class MockSamples:
    def __init__(self, max_log_likelihood_instance=None):

        self.max_log_likelihood_instance = max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma):
        return None


class MockResult:
    def __init__(self, instance, log_likelihood, model=None):
        self.instance = instance
        self.log_likelihood = log_likelihood
        self.model = model
        self.previous_model = model
        self.gaussian_tuples = None
        self.mask_2d = None


class MockResults:
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



class MockNLO(af.NonLinearOptimizer):
    def _simple_fit(self, analysis, fitness_function):
        # noinspection PyTypeChecker
        return af.Result(None, analysis.fit(None), None)

    def _fit(self, analysis, model):
        class Fitness:
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.fit(instance)
                self.result = MockResult(instance=instance)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    def samples_from_model(self, model):
        return MockSamples()