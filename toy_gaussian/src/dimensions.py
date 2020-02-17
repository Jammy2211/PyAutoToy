import autofit as af
import typing

import inspect
from toy_gaussian.src import exc


class DimensionsProfile:
    def __init__(self):

        pass

    def new_object_with_units_converted(
        self,
        unit_length=None,
        unit_luminosity=None,
        kpc_per_arcsec=None,
        exposure_time=None,
    ):

        constructor_args = inspect.getfullargspec(self.__init__).args

        def convert(value):
            if unit_length is not None:
                if isinstance(value, Length):
                    return value.convert(unit_length, kpc_per_arcsec)
                if isinstance(value, tuple):
                    return tuple(convert(item) for item in value)
            if unit_luminosity is not None and isinstance(value, Luminosity):
                return value.convert(unit_luminosity, exposure_time)
            return value

        return self.__class__(
            **{
                key: convert(value)
                for key, value in self.__dict__.items()
                if key in constructor_args
            }
        )

    @property
    def unit_length(self):

        unit_list = []

        for attr, value in self.__dict__.items():

            if isinstance(value, tuple):
                for tuple_value in value:
                    if hasattr(tuple_value, "unit_length"):
                        unit_list.append(tuple_value.unit_length)

            if isinstance(value, float):
                if hasattr(value, "unit_length"):
                    unit_list.append(value.unit_length)

        if len(unit_list) > 0:
            if not all(unit == unit_list[0] for unit in unit_list):
                raise exc.UnitsException(
                    "This object has attributes with different units of length defined"
                )
        else:
            return None

        return unit_list[0]

    @property
    def unit_luminosity(self):

        unit_list = []

        for attr, value in self.__dict__.items():
            if isinstance(value, float):
                if hasattr(value, "unit_luminosity"):
                    unit_list.append(value.unit_luminosity)

        if len(unit_list) > 0:
            if not all(unit == unit_list[0] for unit in unit_list):
                raise exc.UnitsException(
                    "This object has attributes with different unit_label of luminosity defined"
                )
        else:
            return None

        return unit_list[0]


class Length(af.DimensionType):
    def __init__(self, value, unit_length="arcsec"):
        super().__init__(value)
        self.unit_length = unit_length
        self.unit_length_power = 1.0

    @property
    def unit(self):
        return self.unit_length

    def convert(self, unit_length, kpc_per_arcsec=None):
        value = self

        value = convert_length(
            value=value,
            unit_current=self.unit_length,
            unit_new=unit_length,
            power=self.unit_length_power,
            kpc_per_arcsec=kpc_per_arcsec,
        )

        return Length(value=value, unit_length=unit_length)


class Luminosity(af.DimensionType):
    def __init__(self, value, unit_luminosity="eps"):
        super().__init__(value)
        self.unit_luminosity = unit_luminosity
        self.unit_luminosity_power = 1.0

    @property
    def unit(self):
        return self.unit_luminosity

    def convert(self, unit_luminosity, exposure_time=None):
        value = self

        value = convert_luminosity(
            value=value,
            unit_current=self.unit_luminosity,
            unit_new=unit_luminosity,
            power=self.unit_luminosity_power,
            exposure_time=exposure_time,
        )

        return Luminosity(value=value, unit_luminosity=unit_luminosity)


Position = typing.Tuple[Length, Length]


def convert_length(value, unit_current, unit_new, power, kpc_per_arcsec):

    if unit_current not in unit_new and kpc_per_arcsec is None:
        raise exc.UnitsException(
            "The length for a value has been requested in new units without a "
            "kpc_per_arcsec conversion factor."
        )

    if unit_current in unit_new:
        return value
    elif unit_current in "arcsec" and unit_new in "kpc":
        return (kpc_per_arcsec ** power) * value
    elif unit_current in "kpc" and unit_new in "arcsec":
        return value / (kpc_per_arcsec ** power)
    else:
        raise exc.UnitsException(
            "The unit specified for the length of a value was an invalid string, you "
            "must use (arcsec | kpc)"
        )


def convert_luminosity(value, unit_current, unit_new, power, exposure_time):
    if unit_current not in unit_new and exposure_time is None:
        raise exc.UnitsException(
            "The luminosity for a value has been requested in new units "
            "without an  exposure time conversion factor."
        )

    if unit_current in unit_new:
        return value
    elif unit_current in "eps" and unit_new in "counts":
        return (exposure_time ** power) * value
    elif unit_current in "counts" and unit_new in "eps":
        return value / (exposure_time ** power)
    else:
        raise exc.UnitsException(
            "The unit specified for the luminosity of a value was an invalid string, you "
            "must use (electrons per second | counts)"
        )
