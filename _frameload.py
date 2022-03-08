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


def open_img(imgs: str, processes=6) -> np.array:
    with mp.Pool(processes=processes) as pool:
        return np.stack(pool.map(_open_img_mp, imgs))


def _get_label(fname: str) -> float:
    return 1. if 'anime' in fname else 0.


get_label = np.vectorize(_get_label)


def main(train_dir: str, train_suf='**/*.bmp', n_train=60000, n_test=20000) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Load all training and test frames into memory. 
    
    Frames are loaded as PIL objects, then converted to Torch's preferred image format
    as C x H x W (maybe C x W x H, the important thing is that C is first), then cast
    back to numpy for more streamlined (and less error-prone) handling, shuffling, etc...

    Later, in main, this entire N x C x <W x H or H x W> array is
    converted back to a tensor with torch.from_numpy.
    """
    fils = np.array(glob.glob(os.path.join(train_dir, train_suf), recursive=True))  # nopep8
    lbls = get_label(fils)

    if n_train + n_test > len(fils):
        raise ValueError(
            'Asked for more images in combined training and test data than available!')

    shuffled = np.random.permutation(len(fils))
    fils = fils[shuffled]
    lbls = lbls[shuffled]

    fils_trn = fils[0:n_train]
    lbls_trn = lbls[0:n_train]
    fils_tst = fils[n_train:n_train + n_test]
    lbls_tst = lbls[n_train:n_train + n_test]

    imgs_trn = open_img(fils_trn)
    imgs_tst = open_img(fils_tst)

    return torch.from_numpy(imgs_trn), torch.from_numpy(lbls_trn), torch.from_numpy(imgs_tst), torch.from_numpy(lbls_tst)


if __name__ == '__main__':
    out = main('train/')
