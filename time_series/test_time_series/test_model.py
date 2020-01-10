from time_series import lotka_voltera as lv


class TestLotkaVoltera:
    def test_step(self, species_collection):
        model = lv.LotkaVolteraModel(
            species_collection
        )

        assert model.capacity == 1.0
