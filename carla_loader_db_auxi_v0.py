#!/usr/bin/env python
# coding=utf-8
'''
Sequence Data Loader 및 Augmentation
2019.09.10
N 개의 이미지를 불러와서 그 sequence에는 동일한 transform을 적용하는 버전.
초기에 원하는 것은 한 장의 이미지에 N개의 Action GT를 같기를 계획 하였으므로
이 버전은 잠시 보류
'''

import glob

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
from random import randint
from imgaug import augmenters as iaa
from ver0.helper_v0 import RandomTransWrapper, RandomTransWrapper_seqImg

from PIL import Image
import time

class CarlaH5Data():
    def __init__(self,
                 train_folder,
                 eval_folder,
                 num_nt=2,
                 batch_size=4, num_workers=4, distributed=False):

        self.loaders = {
            "train": torch.utils.data.DataLoader(
                CarlaH5Dataset(
                    data_dir=train_folder,
                    train_eval_flag="train", num_nt=num_nt),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_data
            ),
            "eval": torch.utils.data.DataLoader(
                CarlaH5Dataset(
                    data_dir=eval_folder,
                    train_eval_flag="eval", num_nt=num_nt),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
                collate_fn=collate_data
            )}

def collate_data(batch):
    batch = list(filter(lambda x:x is not None, batch))
    # return batch

    imgs = []
    target1 = []
    target2 = []
    target3 = []

    for sample in batch:
        imgs.append(sample[0])
        target1.append(torch.Tensor(sample[1]))
        target2.append(torch.Tensor(sample[2]))
        target3.append(torch.Tensor(sample[3]))

    return torch.stack(imgs, 0), torch.stack(target1, 0), torch.stack(target2, 0), torch.stack(target3, 0)

class CarlaH5Dataset(Dataset):
    def __init__(self, data_dir,
                 train_eval_flag="train", num_nt=2, sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir + '*.h5')
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag
        self.nt = num_nt
        self.img_list = []
        self.rand_prob_list = []
        self.rand_range_list = []

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        # seq=iaa.ContrastNormalization(
                        seq=iaa.LinearContrast(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return self.sequnece_len * len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx // self.sequnece_len
        file_idx = idx % self.sequnece_len
        file_name = self.data_list[data_idx]

        with h5py.File(file_name, 'r') as h5_file:

            lower_idx = file_idx
            if file_idx + self.nt >= self.sequnece_len:
                # lower_idx = self.sequnece_len - self.nt
                return None

            img = np.array(h5_file['rgb'])[lower_idx]
            img = self.transform(img)

            target = np.array(h5_file['targets'])[lower_idx:lower_idx + self.nt]
            target = target.astype(np.float32)

            game_time = target[:self.nt, 20]
            if abs(game_time[self.nt - 1] - game_time[0]) > (self.nt) * 100:
                return None

            command = int(target[0, 24]) - 2
            if command == -2:
                command = 0
            speed = np.array([np.max(target[0, 10], 0) / 40, ]).astype(np.float32)
            # speed = np.array([target[:self.nt, 10] / 40, ]).astype(np.float32)

            target_vec_lateral = np.zeros((4, 1), dtype=np.float32)
            target_vec_lateral[command, :] = target[0, 0]
            mask_vec_lateral = np.zeros((4, 1), dtype=np.float32)
            mask_vec_lateral[command, :] = 1


        return img, speed, target_vec_lateral.reshape(-1), mask_vec_lateral.reshape(-1)


def vector_transform(vec, angle):
    rad_angle = np.deg2rad(angle)
    R = np.array([[np.cos(rad_angle), -np.sin(rad_angle)], [np.sin(rad_angle), np.cos(rad_angle)]])
    return np.inner(R, vec)

def get_predicted_wheel_location(x, y, steering_angle, yaw, v, time_stamp=0.05):
    wheel_heading = np.deg2rad(yaw) + steering_angle
    # wheel_traveled_dis = v * (_current_timestamp - vars.t_previous)
    wheel_traveled_dis = v * time_stamp
    return [x + wheel_traveled_dis * np.cos(wheel_heading), y + wheel_traveled_dis * np.sin(wheel_heading)]

# pred_x is x_t+1
def get_predicted_steering(x, y, pred_x, pred_y, yaw, v, time_stamp=0.05):
    # print((pred_x - x)/(v * 0.05), ' ', (pred_y - y)/(v * 0.05))
    steering_angle_x = np.arccos((pred_x - x)/(v * time_stamp)) - np.deg2rad(yaw)
    steering_angle_y = np.arcsin((pred_y - y)/(v * time_stamp)) - np.deg2rad(yaw)

    return [steering_angle_x, steering_angle_y]

def get_predicted_velocity(x, y, pred_x, pred_y, steering_angle, time_stamp=0.05):
    v_x = (pred_x - x) / (time_stamp * np.cos(steering_angle))
    v_y = (pred_y - y) / (time_stamp * np.sin(steering_angle))

    return [v_x, v_y]

def main():
    train_dir = "/SSD1/datasets/carla/additional_db/gen_data_brake/"
    eval_dir = "/SSD1/datasets/carla/additional_db/val/"
    batch_size = 10
    workers = 1

    carla_data = CarlaH5Data(
        train_folder=train_dir,
        eval_folder=eval_dir,
        num_nt=2,
        batch_size=batch_size,
        num_workers=workers
    )

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]

    start_time = time.time()
    for i, (img, speed, target_vec, mask_vec, target_vec_lat, mask_vec_lat, target_vec_lon, mask_vec_lon, target_vec_trans_x, target_vec_trans_y) in enumerate(train_loader):
        pass
        # print(i, ' : ', img.shape )
        # print("---{}s seconds---".format(time.time()-start_time))
        # start_time = time.time()

if __name__ == '__main__':
    main()

