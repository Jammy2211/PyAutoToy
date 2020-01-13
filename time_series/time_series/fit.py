import numpy as np


class Fit:
    def __init__(self, model_data, observed_data):
        self.model_data = model_data
        self.observed_data = observed_data

    @property
    def residuals(self):
        return self.observed_data - self.model_data

    @property
    def chi_squared_list(self):
        return np.square(
            self.residuals
        )

    @property
    def chi_squared(self):
        return np.sum(self.chi_squared_list)
