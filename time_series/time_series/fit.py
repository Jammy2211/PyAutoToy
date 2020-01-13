class Fit:
    def __init__(self, model_data, observed_data):
        self.model_data = model_data
        self.observed_data = observed_data

    @property
    def residuals(self):
        return self.observed_data - self.model_data
