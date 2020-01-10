import inspect
from collections import defaultdict
from typing import Union

import autofit as af
from time_series.matrix import Matrix


class SpeciesPriorModel(af.PriorModel):
    def __init__(self, cls, **kwargs):
        super().__init__(cls, **kwargs)
        self.interactions = defaultdict(lambda: 0.0)

    def __getitem__(self, species: "SpeciesPriorModel") -> Union[float, af.Prior]:
        """
        Convenience method for getting the interaction value of this species with another.

        If no interaction rate has been set then 1.0 is returned.

        Parameters
        ----------
        species
            A species with which this species interacts

        Returns
        -------
        The negative impact the presence of the other species has on the growth
        of this species.
        """
        return self.interactions[species]

    def __setitem__(
            self,
            species: "SpeciesPriorModel",
            interaction: Union[float, af.Prior]
    ):
        """
        Convenience method for setting the interaction value of this species with another.

        Parameters
        ----------
        species
            A species with which this species interacts
        interaction
            The negative impact the presence of the other species has on the growth
            of this species.
        """
        self.interactions[species] = interaction


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
        super().__init__(list(map(SpeciesPriorModel, items)))
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

    def __len__(self):
        return super().__len__() - 1

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return Matrix.__getitem__(self, item)
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            Matrix.__setitem__(self, key, value)
        else:
            super().__setitem__(key, value)