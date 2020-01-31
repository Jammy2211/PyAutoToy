import toy_gaussian as toy


class TestPipelineGeneralSettings:
    def test__tag(self):

        pipeline_general_settings = toy.PipelineGeneralSettings()

        assert pipeline_general_settings.tag == "pipeline_tag"

        pipeline_general_settings = toy.PipelineGeneralSettings()

        assert pipeline_general_settings.tag == "pipeline_tag"
