from toy_gaussian_workspace.simulate import makers

# Welcome to the PyAutoToy Gaussian test_autoarray suite data maker. Here, we'll make the suite of data_type that we use to test and profile
# PyAutoLens. This consists of the following sets of images:

sub_size = 1

# To simulator each lens, we pass it a name and call its maker. In the makers.py file, you'll see the
makers.make__gaussian(sub_size=sub_size)
