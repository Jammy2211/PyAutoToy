import autoarray as aa


class MetaDatasetFit:
    def __init__(
        self, model, sub_size=2, signal_to_noise_limit=None, is_hyper_phase=False
    ):
        self.is_hyper_phase = is_hyper_phase
        self.model = model
        self.sub_size = sub_size
        self.signal_to_noise_limit = signal_to_noise_limit

    def mask_with_phase_sub_size_from_mask(self, mask):

        if mask.sub_size != self.sub_size:
            mask = aa.mask.manual(
                mask_2d=mask,
                pixel_scales=mask.pixel_scales,
                sub_size=self.sub_size,
                origin=mask.origin,
            )

        return mask
