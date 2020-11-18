import glob
import json
import os
import pickle
import random
from typing import Dict, Optional

import numpy as np
import torch
import tqdm

import constants
from datasets.base_dataset import BatchConcatDataset
from datasets.base_multi_frame_dataset import VideoDataset
from utils import transforms

SAMPLE_NAME = "AA/AA2pFq9pFTA_000001.jpg"
LEN_SAMPLE_NAME = len(SAMPLE_NAME)
LEN_VID_NAME = len("AA2pFq9pFTA")
LEN_NUM_NAME = len("000001")


class Kinetics400Dataset(VideoDataset, BatchConcatDataset):
    @staticmethod
    def get_video_name(name):
        name = name.split('/')
        return os.path.join(name[6], name[7])

    @staticmethod
    def get_frame_id(name):
        img_name, _ = os.path.splitext(name.split('/')[-1])
        return int(img_name[4:])

    def get_image_paths(self):
        image_paths = []
        num_videos = sum([1 for i in open(self.data_list, "r")])
        with open(self.data_list) as txt:
            tbar = tqdm.tqdm(txt, total=num_videos)
            for filename in tbar:
                video_name, _, label = filename.split()
                video_name, _ = os.path.splitext(video_name)
                image_paths += list(glob.iglob(os.path.join(self.data_split_path, video_name, "*.jpg")))
                tbar.set_description("Loading Videos")

        return image_paths

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.data_split_path, key, "img_%05d.jpg" % ind)

    @staticmethod
    def standard_transform(size, data_subset):
        return transforms.Kinetics400Transform(size, data_subset)

    def __init__(
            self,
            args,
            data_subset: str = "train",
            transform=None,
            num_images_to_return=-1,
            shared_transform=True,
            check_for_new_data=False,
    ):
        size = (args.input_height, args.input_width)
        if transform is None:
            transform = self.standard_transform(size, data_subset)

        self.data_list = os.path.join(args.data_path, data_subset + '.txt')

        VideoDataset.__init__(self, args, data_subset, transform, num_images_to_return, check_for_new_data)

        self.shared_transform = shared_transform
        pickle_path = os.path.join(self.data_basepath, self.data_subset + ".pkl")

        if not os.path.exists(pickle_path) or constants.CHECK_FOR_NEW_DATA or check_for_new_data:
            annotations = {}
            with open(self.data_list) as txt:
                for filename in txt:
                    video_name, _, label = filename.split()
                    video_name, _ = os.path.splitext(video_name)
                    annotations[video_name] = int(label)
            pickle.dump(annotations, open(pickle_path, "wb"))

        self.annotations = pickle.load(open(pickle_path, "rb"))

    def __len__(self):
        return len(self.path_info)

    def __getitem__(self, idx) -> Optional[Dict[str, torch.Tensor]]:
        initial_seed = random.randint(0, 2 ** 31)

        path_key, frame_ids = self.path_info[idx]
        start_ind = np.random.randint(1, len(frame_ids) - self.num_images_to_return + 1)

        images = []
        for img_ind in range(start_ind, start_ind + self.num_images_to_return):
            path = self.get_image_name(path_key, img_ind)
            image = self.read_image(path)
            if image is None:
                print("Skipping", path, "missing file")
                return None
            if self.shared_transform:
                self.set_rng(initial_seed)
            image = self.transform(image)
            images.append(image)
        label = self.annotations[path_key]

        return {"data": images, "labels": label, "id": self.path_info[idx], "keys_to_concat": ["data"]}
