import inspect

import autofit as af
from time_series.species import Matrix


class MatrixPriorModel(af.CollectionPriorModel, Matrix):
    def __init__(self, cls: type, items: list):
        """
        A collection prior model with a custom class and convenience
        methods for describing interactions between the members.

        Parameters
        ----------
        cls
            A class that implements the Matrix interface
        items
            A collection of items that implement the Species interface
        """
        super().__init__(items)
        self.cls = cls

    def instance_for_arguments(self, arguments: dict) -> object:
        """
        Create an instance of the class with a ModelInstance created from the set of items
        as the only argument.

        Parameters
        ----------
        arguments
            A dictionary mapping Priors to physical values

        Returns
        -------
        An instance of self.cls
        """
        species = super().instance_for_arguments(
            arguments
        )
        return self.cls(
            list(filter(
                lambda item: not inspect.isclass(item),
                species
            ))
        )

    def __setattr__(self, key, value):
        """
        By default the CollectionPriorModel wraps any attached object in a PriorModel - here
        we want to keep the underlying class object for use in instantiation.
        """
        if key == "cls":
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)
