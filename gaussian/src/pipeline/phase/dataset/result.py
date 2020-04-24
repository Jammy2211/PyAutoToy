from gaussian.src.pipeline.phase.abstract import result


class Result(result.Result):
    @property
    def max_log_likelihood_fit(self):

        return self.analysis.masked_imaging_fit_for_tracer(
            tracer=self.max_log_likelihood_tracer
        )

    @property
    def mask(self):
        return self.max_log_likelihood_fit.mask
