import gaussian as g


class TestPhaseTag:
    def test__mixture_of_values(self):

        phase_tag = g.phase_tagging.phase_tag_from_phase_settings(
            sub_size=2, signal_to_noise_limit=2, bin_up_factor=None
        )

        assert phase_tag == "phase_tag__sub_2__snr_2"

        phase_tag = g.phase_tagging.phase_tag_from_phase_settings(
            sub_size=1, signal_to_noise_limit=None, bin_up_factor=3
        )

        assert phase_tag == "phase_tag__sub_1__bin_3"

        phase_tag = g.phase_tagging.phase_tag_from_phase_settings(
            sub_size=1, real_space_shape_2d=(3, 3), real_space_pixel_scales=(1.0, 2.0)
        )

        assert phase_tag == "phase_tag__rs_shape_3x3__rs_pix_1.00x2.00__sub_1"


class TestPhaseTaggers:
    def test__sub_size_tagger(self):

        tag = g.phase_tagging.sub_size_tag_from_sub_size(sub_size=1)
        assert tag == "__sub_1"
        tag = g.phase_tagging.sub_size_tag_from_sub_size(sub_size=2)
        assert tag == "__sub_2"
        tag = g.phase_tagging.sub_size_tag_from_sub_size(sub_size=4)
        assert tag == "__sub_4"

    def test__signal_to_noise_limit_tagger(self):

        tag = g.phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=None
        )
        assert tag == ""
        tag = g.phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=1
        )
        assert tag == "__snr_1"
        tag = g.phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=2
        )
        assert tag == "__snr_2"
        tag = g.phase_tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=3
        )
        assert tag == "__snr_3"

    def test__bin_up_factor_tagger(self):

        tag = g.phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=None)
        assert tag == ""
        tag = g.phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=1)
        assert tag == ""
        tag = g.phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=2)
        assert tag == "__bin_2"
        tag = g.phase_tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=3)
        assert tag == "__bin_3"

    def test__real_space_shape_2d_tagger(self):

        tag = g.phase_tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=None
        )
        assert tag == ""
        tag = g.phase_tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=(2, 2)
        )
        assert tag == "__rs_shape_2x2"
        tag = g.phase_tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=(3, 4)
        )
        assert tag == "__rs_shape_3x4"

    def test__real_space_pixel_scales_tagger(self):

        tag = g.phase_tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=None
        )
        assert tag == ""
        tag = g.phase_tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=(0.01, 0.02)
        )
        assert tag == "__rs_pix_0.01x0.02"
        tag = g.phase_tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=(2.0, 1.0)
        )
        assert tag == "__rs_pix_2.00x1.00"
