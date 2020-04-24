from gaussian.src.pipeline.phase.dataset import result


class Result(result.Result):
    @property
    def max_log_likelihood_fit(self):

        return self.analysis.masked_imaging_fit_from_instance(instance=self.instance)
