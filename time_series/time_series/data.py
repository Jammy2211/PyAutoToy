import autofit as af


class Data(af.Dataset):
    @property
    def name(self) -> str:
        return "Observables"

    def __init__(
            self,
            **observables
    ):
        self.observables = observables

    @property
    def observable_names(self):
        return set(self.observables.keys())

    def __getitem__(self, observable_name):
        return self.observables[observable_name]

    def __str__(self):
        return str({
            key: value.shape
            for key, value
            in self.observables.items()
        })
