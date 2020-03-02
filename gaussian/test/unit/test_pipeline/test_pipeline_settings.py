import gaussian as g


class TestPipelineGeneralSettings:
    def test__tag(self):

        pipeline_general_settings = g.PipelineGeneralSettings()

        assert pipeline_general_settings.tag == "pipeline_tag"

        pipeline_general_settings = g.PipelineGeneralSettings()

        assert pipeline_general_settings.tag == "pipeline_tag"
