import numpy as np
from scipy import stats


class Observable:
    def __init__(
            self,
            mean,
            deviation
    ):
        self.mean = mean
        self.deviation = deviation

    @property
    def distribution(self):
        return stats.norm(
            loc=self.mean,
            scale=self.deviation
        )

    def pdf(
            self,
            lower_limit=-2,
            upper_limit=2,
            number_points=1000
    ):
        return self.distribution.pdf(
            np.linspace(
                lower_limit,
                upper_limit,
                number_points
            )[:, None]
        )
