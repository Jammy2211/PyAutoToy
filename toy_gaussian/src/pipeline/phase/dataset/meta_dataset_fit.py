import autoarray as aa
from toy_gaussian.src.pipeline.phase.dataset.phase import default_mask_function


class MetaDatasetFit:
    def __init__(
        self,
        model,
        sub_size=2,
        signal_to_noise_limit=None,
        mask_function=None,
        is_hyper_phase=False,
    ):
        self.is_hyper_phase = is_hyper_phase
        self.model = model
        self.sub_size = sub_size
        self.signal_to_noise_limit = signal_to_noise_limit
        self.mask_function = mask_function

    def setup_phase_mask(self, shape_2d, pixel_scales, mask):

        if self.mask_function is not None:
            mask = self.mask_function(shape_2d=shape_2d, pixel_scales=pixel_scales)

        elif mask is None and self.mask_function is None:
            mask = default_mask_function(shape_2d=shape_2d, pixel_scales=pixel_scales)

        if mask.sub_size != self.sub_size:
            mask = aa.mask.manual(
                mask_2d=mask,
                pixel_scales=mask.pixel_scales,
                sub_size=self.sub_size,
                origin=mask.origin,
            )

        return mask
