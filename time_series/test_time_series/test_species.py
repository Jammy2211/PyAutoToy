from time_series import species as s


class TestCase:
    def test_interaction(self):
        species = s.Species()
        assert species.interactions[species] == 1.0

        species.interactions[species] = 0.5
        assert species.interactions[species] == 0.5
