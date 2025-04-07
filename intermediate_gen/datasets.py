import os
import torch 
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

from tqdm import tqdm
from PIL import Image

CIFARC_TYPES = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]

class CIFARDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        transform,
        cifar_type=None,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.cifar_type = cifar_type

        self._load_data()

    def _load_data(self):
        if self.train:
            data = []
            labels = []
            for i in range(1, 6):
                with open(os.path.join(self.root, f"data_batch_{i}"), "rb") as fo:
                    dict = pickle.load(fo, encoding="bytes")
                    data.append(dict[b"data"].reshape(-1, 3, 32, 32))
                    labels.append(dict[b"labels"])
            self.labels = np.concatenate(labels)
            self.data = np.concatenate(data)
            # move dim for channels
            self.data = self.data.transpose(0, 2, 3, 1)

        else:
            if self.cifar_type is None:
                with open(os.path.join(self.root, "test_batch"), "rb") as fo:
                    dict = pickle.load(fo, encoding="bytes")
                    self.data = dict[b"data"].reshape(-1, 3, 32, 32)
                    self.labels = np.array(dict[b"labels"])
                # move dim for channels
                self.data = self.data.transpose(0, 2, 3, 1)
            elif self.cifar_type in CIFARC_TYPES:
                data = []
                labels = []
                # take only the level-5 corruption
                self.labels = np.load(os.path.join(self.root, "labels.npy"))[-10000:]
                self.data = np.load(os.path.join(self.root, f"{self.cifar_type}.npy"))[-10000:]
    def __len__(self) -> int:
        return len(self.data)

    # test_batch

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class MyWaterBirdsDataset(Dataset):
    def __init__(self, basedir, transform=None, remove_minority_groups=False):
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        self.dataset_type = (
            "celeba"
            if "celeba" in basedir.lower()
            else "waterbirds"
            if "waterbirds" in basedir.lower()
            else "unknown"
        )
        print(len(metadata_df))
        self.metadata_df = metadata_df
        print(len(self.metadata_df))
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        self.p_array = self.metadata_df["place"].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df["place"].values
        self.n_places = np.unique(self.confounder_array).size

        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype("int")

        self.n_groups = self.n_classes * self.n_places
        # load the list of attributes - its in csv 'list_attr_celeba.csv'
        
        self.attributes = defaultdict(lambda: 0)
        
        self.group_counts = (
            (torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array))
            .sum(1)
            .float()
        )
        self.y_counts = (
            (torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array))
            .sum(1)
            .float()
        )
        self.p_counts = (
            (torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array))
            .sum(1)
            .float()
        )
        self.filename_array = self.metadata_df["img_filename"].values

        # get train, val and test idxs.
        self.train_idxs = self.metadata_df[self.metadata_df["split"] == 0].index.values
        self.val_idxs = self.metadata_df[self.metadata_df["split"] == 1].index.values
        self.test_idxs = self.metadata_df[self.metadata_df["split"] == 2].index.values

        # if true, for 'waterbirds' remove groups 1, 2; for 'celeba' remove group 3
        self.remove_minority_groups = remove_minority_groups
        if remove_minority_groups:
            # remove only from train_idxs

            if self.dataset_type == "waterbirds":
                # use group 'self.group_array'
                self.train_idxs = self.train_idxs[
                    np.logical_and(
                        self.group_array[self.train_idxs] != 1,
                        self.group_array[self.train_idxs] != 2,
                    )
                ]
            elif self.dataset_type == "celeba":
                # use group 'self.group_array'
                self.train_idxs = self.train_idxs[self.group_array[self.train_idxs] != 3]
            else:
                raise ValueError("Unknown dataset type")

        # Get some basic stats, quickly
        for split_set_name, split_set_idxs in zip(
            ["train", "val", "test"], [self.train_idxs, self.val_idxs, self.test_idxs]
        ):
            print(f"Split set: {split_set_name}")

            # print how many of different y, p exists and differnet group.
            print("y counts", np.bincount(self.y_array[split_set_idxs]))
            print("p counts", np.bincount(self.p_array[split_set_idxs]))
            print("group counts", np.bincount(self.group_array[split_set_idxs]))

        # cache images
        split_into_files = False
        split = "all"
        # self.use_caching = globals.using_cluster and False
        self.use_caching = False
        cahced_name = os.path.join(basedir, f"image_cache_{split}.pkl")
        # use lustre - hack 
        # basedir
        if self.use_caching:
            if not os.path.exists(cahced_name):
                print("Caching images")
                images_cache = []
                cached_second = False

                for i in tqdm(range(len(self.metadata_df))):
                    img_path = os.path.join(self.basedir, self.filename_array[i])
                    img = Image.open(img_path).convert("RGB")
                    # pass image through the transform
                    # img_transformed = self.transform(img)
                    # img_transformed= np.array(img_transformed)

                    # convert to np
                    images_cache.append(img)
                    if split_into_files and i >= (len(self.metadata_df) / 2) and not cached_second:
                        # save it
                        print("Saving images cache")
                        pickle.dump(
                            images_cache,
                            open(os.path.join(basedir, f"image_cache_{split}_2.pkl"), "wb"),
                        )
                        cached_second = True
                        images_cache = []
                # save it
                print("Saving images cache")
                pickle.dump(
                    images_cache,
                    open(os.path.join(basedir, f"image_cache_{split}.pkl"), "wb"),
                )
            else:
                print("Loading images cache")
               
                self.images_cache = pickle.load(open(new_cached_name, "rb"))
                # check if also second file exists - then read again and merge
                if os.path.exists(os.path.join(basedir, f"image_cache_{split}_2.pkl")):
                    self.images_cache_2 = pickle.load(
                        open(os.path.join(basedir, f"image_cache_{split}_2.pkl"), "rb")
                    )
                    # merge
                    self.images_cache_2.extend(self.images_cache)
        self.mapper = self.label_mapper

    def __len__(self):
        return len(self.metadata_df)

    def label_mapper(self, attrs):
        return attrs[0]

    def get_train_group_counts(self):
        return (
            (
                torch.arange(self.n_groups).unsqueeze(1)
                == torch.from_numpy(self.group_array[self.train_idxs])
            )
            .sum(1)
            .float()
        )

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        if not self.use_caching:
            img_path = os.path.join(self.basedir, self.filename_array[idx])
            img = Image.open(img_path).convert("RGB")
        else:
            img = self.images_cache[idx]

        if self.transform:
            img = self.transform(img)
        return (img, y, np.array((y, p)), idx), (g, self.attributes[idx])