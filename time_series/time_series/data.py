class Data:
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
