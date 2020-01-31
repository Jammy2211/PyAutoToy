from toy_gaussian.src import exc
import pytest

import toy_gaussian as toy


class TestLength(object):
    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(
        self
    ):
        unit_arcsec = toy.dim.Length(value=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_length == "arcsec"

        unit_arcsec = unit_arcsec.convert(unit_length="arcsec")

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "arcsec"

        unit_kpc = unit_arcsec.convert(unit_length="kpc", kpc_per_arcsec=2.0)

        assert unit_kpc == 4.0
        assert unit_kpc.unit == "kpc"

        unit_kpc = unit_kpc.convert(unit_length="kpc")

        assert unit_kpc == 4.0
        assert unit_kpc.unit == "kpc"

        unit_arcsec = unit_kpc.convert(unit_length="arcsec", kpc_per_arcsec=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == "arcsec"

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_length="kpc")
            unit_kpc.convert(unit_length="arcsec")
            unit_arcsec.convert(unit_length="lol")


class TestLuminosity(object):
    def test__conversions_from_eps_and_counts_and_back__errors_raised_if_no_exposure_time(
        self
    ):

        unit_eps = toy.dim.Luminosity(value=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit_luminosity == "eps"

        unit_eps = unit_eps.convert(unit_luminosity="eps")

        assert unit_eps == 2.0
        assert unit_eps.unit == "eps"

        unit_counts = unit_eps.convert(unit_luminosity="counts", exposure_time=2.0)

        assert unit_counts == 4.0
        assert unit_counts.unit == "counts"

        unit_counts = unit_counts.convert(unit_luminosity="counts")

        assert unit_counts == 4.0
        assert unit_counts.unit == "counts"

        unit_eps = unit_counts.convert(unit_luminosity="eps", exposure_time=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit == "eps"

        with pytest.raises(exc.UnitsException):
            unit_eps.convert(unit_luminosity="counts")
            unit_counts.convert(unit_luminosity="eps")
            unit_eps.convert(unit_luminosity="lol")


class MockDimensionsProfile(toy.dim.DimensionsProfile):
    def __init__(
        self,
        position: toy.dim.Position = None,
        param_float: float = None,
        length: toy.dim.Length = None,
        luminosity: toy.dim.Luminosity = None,
    ):

        super(MockDimensionsProfile, self).__init__()

        self.position = position
        self.param_float = param_float
        self.luminosity = luminosity
        self.length = length


class TestDimensionsProfile(object):
    class TestUnitProperties(object):
        def test__extracts_length_correctly__raises_error_if_different_lengths_input(
            self
        ):
            profile = MockDimensionsProfile(
                position=(
                    toy.dim.Length(value=3.0, unit_length="arcsec"),
                    toy.dim.Length(value=3.0, unit_length="arcsec"),
                ),
                length=toy.dim.Length(3.0, "arcsec"),
            )

            assert profile.unit_length == "arcsec"

            profile = MockDimensionsProfile(
                position=(
                    toy.dim.Length(value=3.0, unit_length="kpc"),
                    toy.dim.Length(value=3.0, unit_length="kpc"),
                ),
                length=toy.dim.Length(3.0, "kpc"),
            )

            assert profile.unit_length == "kpc"

            with pytest.raises(exc.UnitsException):
                profile = MockDimensionsProfile(
                    position=(
                        toy.dim.Length(value=3.0, unit_length="kpc"),
                        toy.dim.Length(value=3.0, unit_length="kpc"),
                    ),
                    length=toy.dim.Length(3.0, "arcsec"),
                )

                profile.unit_length

        def test__extracts_luminosity_correctly__raises_error_if_different_luminosities(
            self
        ):
            profile = MockDimensionsProfile(luminosity=toy.dim.Luminosity(3.0, "eps"))

            assert profile.unit_luminosity == "eps"

            profile = MockDimensionsProfile(
                luminosity=toy.dim.Luminosity(3.0, "counts")
            )

            assert profile.unit_luminosity == "counts"

    class TestUnitConversions(object):
        def test__arcsec_to_kpc_conversions_of_length__float_and_tuple_length__conversion_converts_values(
            self
        ):

            profile_arcsec = MockDimensionsProfile(
                position=(toy.dim.Length(1.0, "arcsec"), toy.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=toy.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=toy.dim.Luminosity(value=4.0, unit_luminosity="eps"),
            )

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit_length == "arcsec"
            assert profile_arcsec.position[1].unit_length == "arcsec"
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit_length == "arcsec"
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit_luminosity == "eps"

            profile_arcsec = profile_arcsec.new_object_with_units_converted(
                unit_length="arcsec"
            )

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit == "arcsec"
            assert profile_arcsec.position[1].unit == "arcsec"
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit == "arcsec"
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit == "eps"

            profile_kpc = profile_arcsec.new_object_with_units_converted(
                unit_length="kpc", kpc_per_arcsec=2.0
            )

            assert profile_kpc.position == (2.0, 4.0)
            assert profile_kpc.position[0].unit == "kpc"
            assert profile_kpc.position[1].unit == "kpc"
            assert profile_kpc.param_float == 2.0
            assert profile_kpc.length == 6.0
            assert profile_kpc.length.unit == "kpc"
            assert profile_kpc.luminosity == 4.0
            assert profile_kpc.luminosity.unit == "eps"

            profile_kpc = profile_kpc.new_object_with_units_converted(unit_length="kpc")

            assert profile_kpc.position == (2.0, 4.0)
            assert profile_kpc.position[0].unit == "kpc"
            assert profile_kpc.position[1].unit == "kpc"
            assert profile_kpc.param_float == 2.0
            assert profile_kpc.length == 6.0
            assert profile_kpc.length.unit == "kpc"
            assert profile_kpc.luminosity == 4.0
            assert profile_kpc.luminosity.unit == "eps"

            profile_arcsec = profile_kpc.new_object_with_units_converted(
                unit_length="arcsec", kpc_per_arcsec=2.0
            )

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit == "arcsec"
            assert profile_arcsec.position[1].unit == "arcsec"
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit == "arcsec"
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit == "eps"

        def test__conversion_requires_kpc_per_arcsec_but_does_not_supply_it_raises_error(
            self
        ):

            profile_arcsec = MockDimensionsProfile(
                position=(toy.dim.Length(1.0, "arcsec"), toy.dim.Length(2.0, "arcsec"))
            )

            with pytest.raises(exc.UnitsException):
                profile_arcsec.new_object_with_units_converted(unit_length="kpc")

            profile_kpc = profile_arcsec.new_object_with_units_converted(
                unit_length="kpc", kpc_per_arcsec=2.0
            )

            with pytest.raises(exc.UnitsException):
                profile_kpc.new_object_with_units_converted(unit_length="arcsec")

        def test__eps_to_counts_conversions_of_luminosity__conversions_convert_values(
            self
        ):

            profile_eps = MockDimensionsProfile(
                position=(toy.dim.Length(1.0, "arcsec"), toy.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=toy.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=toy.dim.Luminosity(value=4.0, unit_luminosity="eps"),
            )

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == "arcsec"
            assert profile_eps.position[1].unit_length == "arcsec"
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == "arcsec"
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == "eps"

            profile_eps = profile_eps.new_object_with_units_converted(
                unit_luminosity="eps"
            )

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == "arcsec"
            assert profile_eps.position[1].unit_length == "arcsec"
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == "arcsec"
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == "eps"

            profile_counts = profile_eps.new_object_with_units_converted(
                unit_luminosity="counts", exposure_time=10.0
            )

            assert profile_counts.position == (1.0, 2.0)
            assert profile_counts.position[0].unit_length == "arcsec"
            assert profile_counts.position[1].unit_length == "arcsec"
            assert profile_counts.param_float == 2.0
            assert profile_counts.length == 3.0
            assert profile_counts.length.unit_length == "arcsec"
            assert profile_counts.luminosity == 40.0
            assert profile_counts.luminosity.unit_luminosity == "counts"

            profile_counts = profile_counts.new_object_with_units_converted(
                unit_luminosity="counts"
            )

            assert profile_counts.position == (1.0, 2.0)
            assert profile_counts.position[0].unit_length == "arcsec"
            assert profile_counts.position[1].unit_length == "arcsec"
            assert profile_counts.param_float == 2.0
            assert profile_counts.length == 3.0
            assert profile_counts.length.unit_length == "arcsec"
            assert profile_counts.luminosity == 40.0
            assert profile_counts.luminosity.unit_luminosity == "counts"

            profile_eps = profile_counts.new_object_with_units_converted(
                unit_luminosity="eps", exposure_time=10.0
            )

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == "arcsec"
            assert profile_eps.position[1].unit_length == "arcsec"
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == "arcsec"
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == "eps"

        def test__luminosity_conversion_requires_exposure_time_but_does_not_supply_it_raises_error(
            self
        ):

            profile_eps = MockDimensionsProfile(
                position=(toy.dim.Length(1.0, "arcsec"), toy.dim.Length(2.0, "arcsec")),
                param_float=2.0,
                length=toy.dim.Length(value=3.0, unit_length="arcsec"),
                luminosity=toy.dim.Luminosity(value=4.0, unit_luminosity="eps"),
            )

            with pytest.raises(exc.UnitsException):
                profile_eps.new_object_with_units_converted(unit_luminosity="counts")

            profile_counts = profile_eps.new_object_with_units_converted(
                unit_luminosity="counts", exposure_time=10.0
            )

            with pytest.raises(exc.UnitsException):
                profile_counts.new_object_with_units_converted(unit_luminosity="eps")
