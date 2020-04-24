from autoarray.dataset.imaging import MaskedImaging
from gaussian.src.pipeline.phase.dataset import meta_dataset


class MetaImaging(meta_dataset.MetaDataset):
    def __init__(
        self, model, sub_size=2, signal_to_noise_limit=None, bin_up_factor=None
    ):
        super().__init__(
            model=model, sub_size=sub_size, signal_to_noise_limit=signal_to_noise_limit
        )
        self.bin_up_factor = bin_up_factor

    def masked_dataset_from(self, dataset, mask, results):


        mask = self.mask_with_phase_sub_size_from_mask(mask=mask)

        if self.bin_up_factor is not None:

            dataset = dataset.binned_from_bin_up_factor(
                bin_up_factor=self.bin_up_factor
            )

            mask = mask.mapping.binned_mask_from_bin_up_factor(
                bin_up_factor=self.bin_up_factor
            )

        if self.signal_to_noise_limit is not None:
            dataset = dataset.signal_to_noise_limited_from_signal_to_noise_limit(
                signal_to_noise_limit=self.signal_to_noise_limit
            )

        return MaskedImaging(
            imaging=dataset, mask=mask
        )