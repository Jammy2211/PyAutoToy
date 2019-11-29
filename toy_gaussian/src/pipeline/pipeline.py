import autofit as af


class PipelineSettings(object):
    def __init__(self,):

        pass


class PipelineDataset(af.Pipeline):
    def __init__(self, pipeline_name, pipeline_tag, *phases, hyper_mode=False):

        super(PipelineDataset, self).__init__(pipeline_name, pipeline_tag, *phases)

        self.hyper_mode = hyper_mode

    def run(self, dataset, mask=None, data_name=None):
        def runner(phase, results):
            return phase.run(dataset=dataset, results=results, mask=mask)

        return self.run_function(runner, data_name)
