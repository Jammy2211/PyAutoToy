from autoarray.plot import mat_objs


class GaussianCentreScatterer(mat_objs.Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):

        super(GaussianCentreScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="gaussian_centres",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return GaussianCentreScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )
