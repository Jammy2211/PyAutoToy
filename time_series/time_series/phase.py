import autofit as af


class SingleTimePhase(af.Phase):
    @af.convert_paths
    def __init__(
            self,
            paths,
            data_index,
            *,
            analysis_class,
            optimizer_class=af.MultiNest,
            model=None,
    ):
        paths.phase_name = f"{paths.phase_name}_{data_index}"
        self.data_index = data_index
        super().__init__(
            paths,
            analysis_class=analysis_class,
            optimizer_class=optimizer_class,
            model=model
        )

    def make_analysis(self, dataset):
        return self.analysis_class(
            dataset[self.data_index]
        )
