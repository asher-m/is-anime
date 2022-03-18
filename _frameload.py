import glob
import multiprocessing as mp
import numpy as np
import os

import PIL

import torch
import torchvision


totensor = torchvision.transforms.ToTensor()


def _open_img_mp(img):
    with PIL.Image.open(img) as pilimg:
        return totensor(pilimg).numpy()


def open_img(imgs: str, processes) -> np.array:
    with mp.Pool(processes=processes) as pool:
        return np.stack(pool.map(_open_img_mp, imgs))


def _get_label(fname: str) -> float:
    return 1. if 'anime' in fname else 0.


get_label = np.vectorize(_get_label)


def main(train_dir: str, train_suf='**/*.jpg', n_train=4000, n_test=1000, processes=8) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Load all training and test frames into memory. 

    Frames are loaded as PIL objects, then converted to Torch's preferred image format
    as C x H x W (maybe C x W x H, the important thing is that C is first), then cast
    back to numpy for more streamlined (and less error-prone) handling, shuffling, etc...

    Later, in main, this entire N x C x <W x H or H x W> array is
    converted back to a tensor with torch.from_numpy.
    """
    assert n_train % 2 == 0  # required (for this script) to produce a balanced dataset

    _files_photo = np.array(glob.glob(os.path.join(train_dir, 'photo', train_suf), recursive=True))  # nopep8
    _files_anime = np.array(glob.glob(os.path.join(train_dir, 'anime', train_suf), recursive=True))  # nopep8
    np.random.shuffle(_files_photo)
    np.random.shuffle(_files_anime)
    _labels_photo = get_label(_files_photo)
    _labels_anime = get_label(_files_anime)

    if n_train / 2 + n_test / 2 > len(_files_photo) or n_train / 2 + n_test / 2 > len(_files_anime):
        raise ValueError(
            'Asked for more images in combined training and test data than available!')

    # get indicies of where to look for files/images so we have a balanced dataset
    train_idx = n_train // 2
    test_idx = (n_train + n_test) // 2

    # concat photo and anime files/images and labels for each dataset
    _files_train = np.concatenate([_files_photo[:train_idx], _files_anime[:train_idx]])  # nopep8
    _files_test = np.concatenate([_files_photo[train_idx:test_idx], _files_anime[train_idx:test_idx]])  # nopep8
    labels_train = np.concatenate([_labels_photo[:train_idx], _labels_anime[:train_idx]])  # nopep8
    labels_test = np.concatenate([_labels_photo[train_idx:test_idx], _labels_anime[train_idx:test_idx]])  # nopep8

    # shuffle train dataset
    _shuffled_train = np.random.permutation(len(_files_train))
    _files_train = _files_train[_shuffled_train]
    labels_train = labels_train[_shuffled_train]

    # shuffle test dataset
    _shuffled_test = np.random.permutation(len(_files_test))
    _files_test = _files_test[_shuffled_test]
    labels_test = labels_test[_shuffled_test]

    # open images
    images_train = open_img(_files_train, processes=processes)
    images_test = open_img(_files_test, processes=processes)

    return torch.from_numpy(images_train), torch.from_numpy(labels_train), torch.from_numpy(images_test), torch.from_numpy(labels_test)


if __name__ == '__main__':
    out = main('train/')
