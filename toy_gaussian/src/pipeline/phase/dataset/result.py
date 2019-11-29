from toy_gaussian.src.pipeline.phase.abstract import result


class Result(result.Result):
    @property
    def most_likely_fit(self):

        return self.analysis.masked_imaging_fit_for_tracer(
            tracer=self.most_likely_tracer
        )

    @property
    def mask(self):
        return self.most_likely_fit.mask
