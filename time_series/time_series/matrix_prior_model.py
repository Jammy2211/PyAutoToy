import inspect

import autofit as af
from time_series.species import Matrix


class MatrixPriorModel(af.CollectionPriorModel, Matrix):
    def __init__(self, cls, *items):
        super().__init__(*items)
        self.cls = cls

    def instance_for_arguments(self, arguments):
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
        if key == "cls":
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)
