import toy_gaussian as toy


class TestPipelineNameTag:
    def test__pipeline_tag__mixture_of_values(self):
        pipeline_tag = toy.pipeline_tagging.pipeline_tag_from_pipeline_settings()

        assert pipeline_tag == "pipeline_tag"