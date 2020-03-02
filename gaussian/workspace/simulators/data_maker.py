from gaussian.workspace.simulators import makers

# Welcome to the PyAutoToy Gaussian test_autoarray suite data maker. Here, we'll make the suite of data_type that we use to test and profile
# PyAutoLens. This consists of the following sets of images:

sub_size = 1

# To simulator each lens, we pass it a name and call its maker. In the makers.py file, you'll see the
makers.make__gaussian_x1(sub_size=sub_size)
makers.make__gaussian_x2(sub_size=sub_size)
makers.make__gaussian_x2_aligned(sub_size=sub_size)
makers.make__gaussian__sub_gaussian(sub_size=sub_size)
[
    makers.make__gaussian_x1__input_sigma(sub_size=sub_size, sigma=sigma)
    for sigma in [0.1, 0.5, 1.0]
]
