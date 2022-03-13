import numpy as np


def many_logical_and(*args):
    """ `logical and` together many arrays without the need to nest the function grossly. """
    if len(args) > 1:
        return np.logical_and(args[0], many_logical_and(args[1:]))
    else:
        return args[-1]


def maskLowDeviationRGB(images, cutoff=0.1):
    """ Takes a torch.Tensor and masks out images likely to be bad samples based on RGB deviation. 

    RGB deviation is defined here as strictly the mean
    of the standard deviations of the RGB channels.
    This round-about definition was created to capture
    and exclude images that may be mostly monochrome, even
    if not dark.

    Expects images to be a torch.Tensor of shape [n_images, C, W, H] (or [..., H, W]).
    """
    lcdev = np.mean(np.std(images.numpy(), axis=(2, 3)), axis=(1,))  # should be of shape [n_images,]
    return lcdev > np.percentile(lcdev, cutoff * 100.)  # now mask arr of shape [n_images,]


def maskLowMeanRGB(images, cutoff=0.1):
    """ Takes a torch.Tensor and masks out images likely to be bad samples based on mean RGB value (brightness).

    Expects images to be a torch.Tensor of shape [n_images, C, W, H] (or [..., H, W]).
    """
    lmdev = np.mean(images.numpy(), axis=(1, 2, 3,))
    return lmdev > np.percentile(lmdev, cutoff * 100.)


def main(images):
    """ Takes a torch.Tensor and masks out images likely to be bad samples.

    Expects images to be a torch.Tensor of shape [n_images, C, W, H] (or [..., H, W]).
    """

    return many_logical_and(*[
        maskLowDeviationRGB(images),
        maskLowMeanRGB(images)
    ])
