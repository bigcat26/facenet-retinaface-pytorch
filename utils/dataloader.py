
import os
import random
import numpy as np
import torch
import torchvision.datasets as datasets
from io import BytesIO
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input, resize_image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class FacenetDataset(Dataset):
    def __init__(self, input_shape, env, index, classes, random):
        self.input_shape    = input_shape
        self.env            = env
        self.index          = index
        self.classes        = classes
        self.random         = random
        self.num_classes    = len(classes)
        # 将index转为记录行
        lines = []
        for i in index:
            for j in index[i]:
                lines.append(f'{i}/{j}')
        self.lines = lines
        #------------------------------------#
        #   路径和标签
        #------------------------------------#
        # self.paths  = []
        # self.labels = []

        # self.load_dataset()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        #------------------------------------#
        #   创建全为零的矩阵
        #------------------------------------#
        images = np.zeros((3, 3, self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((3))

        # 拿到第index条记录
        if self.random:
            item = self.lines[random.randint(0, len(self.lines) - 1)]
        else:
            item = self.lines[index]

        key = None
        while key is None:
            # 分解出class和index
            parts = item.split('/')
            key = parts[0]
            val = parts[1]

            if len(self.index[key]) < 2:
                # 这条记录不满2个, 随机换一条
                key = None
                item = self.lines[random.randint(0, len(self.lines) - 1)]

        positive = val
        while positive == val:
            # 随机找出另一个不相同的图
            positive = self.index[key][random.randint(0, len(self.index[key]) - 1)]

        # 随机找出另一个不相同的class的任意图
        negative_key = key
        while negative_key == key or negative_key is None:
            # 验证集分类无法覆盖训练集时, 有些class无法获取到, 需要跳过处理
            negative_key = self.classes[random.randint(0, len(self.classes) - 1)]
            if self.index.get(negative_key) is None:
                continue
            if len(self.index[negative_key]) == 0:
                negative_key = None
                continue
            negative = self.index[negative_key][random.randint(0, len(self.index[negative_key]) - 1)]

        labels[0] = self.classes.index(key)
        labels[1] = self.classes.index(key)
        labels[2] = self.classes.index(negative_key)

        with self.env.begin(write=False) as txn:
            # origin
            dat = txn.get(f'{key}/{val}'.encode('utf-8'))
            if dat is None:
                raise(f'unable to read key: {key}/{val}')
            image = cvtColor(Image.open(BytesIO(dat)))
            if image is None:
                raise(f'unable to read image: {key}/{val}')
            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            image = np.transpose(image, [2, 0, 1])
            images[0, :, :, :] = image

            # positive
            dat = txn.get(f'{key}/{positive}'.encode('utf-8'))
            if dat is None:
                raise(f'unable to read key: {key}/{val}')
            image = cvtColor(Image.open(BytesIO(dat)))
            if image is None:
                raise(f'unable to read image: {key}/{val}')
            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            image = np.transpose(image, [2, 0, 1])
            images[1, :, :, :] = image

            # negative
            dat = txn.get(f'{negative_key}/{negative}'.encode('utf-8'))
            if dat is None:
                raise(f'unable to read key: {key}/{val}')
            image = cvtColor(Image.open(BytesIO(dat)))
            if image is None:
                raise(f'unable to read image: {key}/{val}')
            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            image = np.transpose(image, [2, 0, 1])
            images[2, :, :, :] = image

        return images, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
        
# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)
    
    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels  = torch.from_numpy(np.array(labels)).long()
    return images, labels

class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir,transform)
        self.image_size = image_size
        self.pairs_path = pairs_path
        self.validation_images = self.get_lfw_paths(dir)

    def read_lfw_pairs(self,pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self,lfw_dir,file_ext="jpg"):

        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in range(len(pairs)):
        #for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame)    = self.validation_images[index]
        image1, image2              = Image.open(path_1), Image.open(path_2)

        image1 = resize_image(image1, [self.image_size[1], self.image_size[0]], letterbox = True)
        image2 = resize_image(image2, [self.image_size[1], self.image_size[0]], letterbox = True)
        
        image1, image2 = np.transpose(preprocess_input(np.array(image1, np.float32)),[2, 0, 1]), np.transpose(preprocess_input(np.array(image2, np.float32)),[2, 0, 1])

        return image1, image2, issame

    def __len__(self):
        return len(self.validation_images)
