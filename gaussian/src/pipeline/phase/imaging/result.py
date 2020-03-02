from gaussian.src.pipeline.phase.dataset import result


class Result(result.Result):
    @property
    def most_likely_fit(self):

        return self.analysis.masked_imaging_fit_from_instance(instance=self.instance)
