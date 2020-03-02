import logging
from typing import Union

import autofit as af
from autofit import ModelObject
from time_series.matrix import Matrix, Species

logger = logging.getLogger(__name__)


class SpeciesPriorModel(af.PriorModel):
    def __init__(self, cls=Species, **kwargs):
        """
        Prior model for a species in a matrix that has defined relationships with other species.

        Parameters
        ----------
        cls
            The class of the species
        kwargs
        """
        super().__init__(cls, **kwargs)
        self.interactions = af.CollectionPriorModel()

    def __str__(self):
        return f"{self.__class__.__name__} {self.id}"

    def instance_for_arguments(self, arguments: {ModelObject: object}):
        """
        Create an instance of the species class with interactions given actual values from priors.

        Parameters
        ----------
        arguments
            Maps prior objects to associated physical values.

        Returns
        -------
        An instance of this species.
        """
        arguments["interactions"] = self.interactions.instance_for_arguments(arguments)
        return super().instance_for_arguments(arguments)

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
        self, species: "SpeciesPriorModel", interaction: Union[float, af.Prior]
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
        species = [
            s
            for s in super().instance_for_arguments(arguments)
            if isinstance(s, Species)
        ]
        pair_map = {
            str(model): instance
            for model, instance in zip(self, species)
            if isinstance(model, SpeciesPriorModel)
        }

        for s in species:
            s.interactions = {
                pair_map[model]: value
                for model, value in s.interactions.items()
                if model in pair_map
            }

        return self.cls(species)

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
