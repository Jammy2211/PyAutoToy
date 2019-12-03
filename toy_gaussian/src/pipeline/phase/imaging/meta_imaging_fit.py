from autoarray.masked import masked_dataset
from toy_gaussian.src.pipeline.phase.dataset import meta_dataset_fit


class MetaImagingFit(meta_dataset_fit.MetaDatasetFit):
    def __init__(
        self,
        model,
        sub_size=2,
        signal_to_noise_limit=None,
        mask_function=None,
        bin_up_factor=None,
    ):
        super().__init__(
            model=model,
            sub_size=sub_size,
            signal_to_noise_limit=signal_to_noise_limit,
            mask_function=mask_function,
        )
        self.bin_up_factor = bin_up_factor

    def masked_dataset_from(self, dataset, mask, results, modified_image):
        mask = self.setup_phase_mask(
            shape_2d=dataset.shape_2d, pixel_scales=dataset.pixel_scales, mask=mask
        )

        masked_imaging = masked_dataset.MaskedImaging(
            imaging=dataset.modified_image_from_image(modified_image), mask=mask
        )

        if self.signal_to_noise_limit is not None:
            masked_imaging = masked_imaging.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        if self.bin_up_factor is not None:
            masked_imaging = masked_imaging.binned_from_bin_up_factor(
                bin_up_factor=self.bin_up_factor
            )

        return masked_imaging
