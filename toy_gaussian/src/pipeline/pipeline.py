import autofit as af


class PipelineSettings:
    def __init__(self,):

        pass


class PipelineDataset(af.Pipeline):
    def __init__(self, pipeline_name, pipeline_tag, *phases):

        super(PipelineDataset, self).__init__(pipeline_name, pipeline_tag, *phases)

    def run(self, dataset, mask=None, data_name=None):
        def runner(phase, results):
            return phase.run(dataset=dataset, results=results, mask=mask)

        return self.run_function(runner, data_name)
