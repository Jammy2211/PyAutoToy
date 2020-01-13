import numpy as np
from scipy import stats


class Observable:
    def __init__(
            self,
            mean: float,
            deviation: float
    ):
        """
        An observable feature associated with one species which has a Gaussian distribution for that species.

        Parameters
        ----------
        mean
            The mean of the distribution
        deviation
            The standard deviation of the distribution
        """
        self.mean = mean
        self.deviation = deviation

    @property
    def distribution(self):
        """
        A function for sampling a normal distribution
        """
        return stats.norm(
            loc=self.mean,
            scale=self.deviation
        )

    def pdf(
            self,
            lower_limit: int = -2,
            upper_limit: int = 2,
            number_points: int = 1000
    ) -> np.ndarray:
        """
        Generate a numpy array describing the point density function with a given number of points
        between two limits.

        Parameters
        ----------
        lower_limit
        upper_limit
        number_points

        Returns
        -------
        An array illustrating the point density
        """
        return self.distribution.pdf(
            np.linspace(
                lower_limit,
                upper_limit,
                number_points
            )[:, None]
        )
