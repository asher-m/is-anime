import glob
import multiprocessing as mp
import numpy as np
import os

import PIL

import torch
import torchvision


totensor = torchvision.transforms.ToTensor()
topilimage = torchvision.transforms.ToPILImage()
transform_forward = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(384),
    torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),
])
transform_backward = torchvision.transforms.Compose([
    torchvision.transforms.Normalize([-1.] * 3, [2.] * 3),
    torchvision.transforms.ToPILImage(),
])


get_label = np.vectorize(lambda fname: 1. if 'anime' in fname else 0.)


def _open_img_mp(img: str):
    with PIL.Image.open(img) as pilimg:
        return totensor(pilimg)


def get_image(imgs: str, processes=6) -> np.array:
    with mp.Pool(processes=processes) as pool:
        return torch.stack(pool.map(_open_img_mp, imgs))


def get_files(train_dir: str, train_suf: str):
    _files_photo = np.array(glob.glob(os.path.join(train_dir, 'photo', train_suf), recursive=True))  # nopep8
    _files_anime = np.array(glob.glob(os.path.join(train_dir, 'anime', train_suf), recursive=True))  # nopep8
    np.random.shuffle(_files_photo)
    np.random.shuffle(_files_anime)

    return _files_photo, _files_anime


class FrameFiles:
    def __init__(self, train_dir='./train/', train_suf='**/*.jpg'):
        # get all available files
        self._files_photo, self._files_anime = get_files(train_dir, train_suf)

        # initialize masks to not repick already allocateed files for other datasets
        self._used_photo = np.zeros_like(self._files_photo, dtype=bool)
        self._used_anime = np.zeros_like(self._files_anime, dtype=bool)

        # initialize indices, nicer so we don't have to do this work many times
        self._indices_photo = np.arange(len(self._files_photo))
        self._indices_anime = np.arange(len(self._files_anime))

        # dict to track uses of files; used to allocate/deallocate files when reloaded
        self._allocated = dict()

    def allocate(self, nfiles, name):
        if not nfiles % 2 == 0:
            raise ValueError(
                'nfiles must be divisible by 2 to create balanced dataset!')

        # pick new files from those that aren't used right now
        picked_photo = np.random.choice(self._indices_photo[~self._used_photo], nfiles // 2, replace=False)  # nopep8
        picked_anime = np.random.choice(self._indices_anime[~self._used_anime], nfiles // 2, replace=False)  # nopep8

        # record what we're using and what we're using it for
        self._used_photo[picked_photo] = True
        self._used_anime[picked_anime] = True

        # get what we just had allocated to this name
        if name in self._allocated:
            allocatement_old = self._allocated.pop(name)
            self._used_photo[allocatement_old['photo']] = False
            self._used_anime[allocatement_old['anime']] = False

        # record what we're using
        self._allocated[name] = dict(photo=picked_photo, anime=picked_anime)

        return self._files_photo[picked_photo], self._files_anime[picked_anime]

    def deallocate(self, name):
        # get what we just had allocated to this name
        if name in self._allocated:
            allocatement_old = self._allocated.pop(name)
            self._used_photo[allocatement_old['photo']] = False
            self._used_anime[allocatement_old['anime']] = False


class Frame(torch.utils.data.Dataset):
    def __init__(self, files: FrameFiles, name: str, nfiles=4000, transform=transform_forward):
        self.files = files
        self.nfiles = nfiles
        self.name = name
        self.transform = transform
        self.transform_target = None

        # allocate some files from the underlying files
        self._files_photo, self._files_anime = self.files.allocate(
            self.nfiles, self.name)
        # label them, turn them into something usable, then mix them
        shuffled = np.random.permutation(len(self._files_anime) + len(self._files_photo))  # nopep8
        self.labels = torch.concat(
            [torch.from_numpy(get_label(self._files_photo)), torch.from_numpy(get_label(self._files_anime))])[shuffled].float()
        self.images = torch.concat(
            [get_image(self._files_photo), get_image(self._files_anime)])[shuffled].float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.transform_target:
            label = self.transform_target(label)

        return image, label

    def __del__(self):
        self.files.deallocate(self.name)

    def reload(self):
        # cleanup old stuff, like make sure it's really, really gone
        del self.images
        del self.labels

        # allocate some files from the underlying files
        self._files_photo, self._files_anime = self.files.allocate(
            self.nfiles, self.name)
        # label them, turn them into something usable, then mix them
        shuffled = np.random.permutation(len(self._files_anime) + len(self._files_photo))  # nopep8
        self.labels = torch.concat(
            [torch.from_numpy(get_label(self._files_photo)), torch.from_numpy(get_label(self._files_anime))])[shuffled].float()
        self.images = torch.concat(
            [get_image(self._files_photo), get_image(self._files_anime)])[shuffled].float()

    def refresh(self):
        # alias for self.reload
        self.reload()
