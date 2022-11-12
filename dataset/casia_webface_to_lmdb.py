#!/usr/bin/env python3
import os
import sys
import time
import math
import gzip
import argparse
import pickle
import random
from io import BytesIO
from PIL import Image
from functools import reduce
from tqdm import tqdm

import lmdb

def scan_dataset(root: str, exts=None, limit=sys.maxsize):
    index = {}
    if exts is None:
        exts = ['jpg', 'png']

    while root.endswith('/'):
        root = root[:len(root) - 1]

    with tqdm(total=1) as pbar:
        for root, _, files in os.walk(root):
            pbar.total += len(files)
            pbar.refresh()

            for name in files:
                match = reduce((lambda x, y: x or y), [
                                name.endswith(ext) for ext in exts])
                if not match:
                    pbar.total -= 1
                    print(f'file {name} skipped')
                    continue
                label = os.path.basename(root)
                if index.get(label) is None:
                    index[label] = []
                index[label].append(name)
                pbar.update(1)

                if pbar.n >= limit:
                    # partial scan complete
                    return (index, pbar.n)

    # scan complete
    return (index, pbar.n)

def casia_webface_to_lmdb(root: str, index: dict, database: str):
    """
    read files in root, store images with path into database
    key value format: | root/file | file data |
    """
    files = []
    for i in index:
        for j in index[i]:
            files.append(f'{i}/{j}')

    env = lmdb.open(database, map_size=1099511627776)
    with tqdm(total=len(files)) as pbar:
        with env.begin(write=True) as txn:
            for i, key in enumerate(files):
                file = f'{root}/{key}'
                with open(file, 'rb') as f:
                    txn.put(key.encode('utf-8'), f.read())
                # pbar.total += 1
                # pbar.refresh()
                pbar.update(1)
    print('done!')

def save_indexes(index):
    # 获取总量
    total = 0
    for k in index:
        total += len(index[k])
    
    # 计算val和train分别占多少份
    num_val = 10575 * 2 # math.ceil(total * 0.01)
    num_train = total - num_val
    
    # 将val份分摊到每个class上，每个class最少需要贡献几份?
    # val_per_class = num_val // len(index)

    # if val_per_class == 0:
    #     # 一个class还分不到一张val，占比太小了
    #     raise(f'val set size ({num_val}) is too small!')
    
    print(f'train set size: {num_train}, val set size: {num_val}, total: {total}')

    # 从各个class抽取val, 直到抽完
    i = 0
    val_size = 0
    val_index = {}
    keys = list(index.keys())
    random.seed(time.time())
    while val_size < num_val:
        i = val_size % len(keys)
        select = random.randint(0, len(index[keys[i]]) - 1)
        if val_index.get(keys[i]) is None:
            val_index[keys[i]] = []
        val_index[keys[i]].append(index[keys[i]][select])
        index[keys[i]].pop(select)
        val_size = val_size + 1

    with gzip.open('classes.gz', 'wb') as f:
        f.write(pickle.dumps(keys))

    with gzip.open('train-index.gz', 'wb') as f:
        f.write(pickle.dumps(index))

    with gzip.open('val-index.gz', 'wb') as f:
        f.write(pickle.dumps(val_index))

    print('done')

def count_index(index):
    size = 0
    for i in index:
        size += len(index[i])
    return size

if __name__ == '__main__':

    # index, n = scan_dataset('/mnt/dataset/CASIA-WebFaces/datasets')
    # with gzip.open('index.gz', 'wb') as f:
    #     f.write(pickle.dumps(index))

    # with gzip.open('index.gz', 'rb') as f:
    #     index = pickle.load(f)
    #     print(f'total classes: {len(index)}')
    # save_indexes(index)

    # with gzip.open('index.gz', 'rb') as f:
    #     #index = pickle.load(f)
    #     print(f'index size={count_index(pickle.load(f))}')

    with gzip.open('train-index.gz', 'rb') as f:
        index_train = pickle.load(f)
        for i in index_train:
            if len(index_train[i]) < 2:
                print(index_train[i])
        print(f'train size={count_index(index_train)}')

    with gzip.open('val-index.gz', 'rb') as f:
        index_val = pickle.load(f)
        for i in index_val:
            if not len(index_val[i]) == 2:
                print(index_val[i])
        print(f'val size={count_index(index_val)}')
        # some = index_val.get('1367048')
        # print(f'some = {some}')


    # casia_webface_to_lmdb('/mnt/dataset/CASIA-WebFaces/datasets', index, '/mnt/dataset/CASIA-WebFaces/database')

    # with open('img/zhangxueyou.jpg', 'rb') as f:
    #     dat = f.read()

    # image = Image.open(BytesIO(dat))
    # image.save('zhang2.png')
    

    
    # parser = argparse.ArgumentParser(description='CASIA WebFace dataset to LMDB converter')
    # parser.add_argument('-i', '--input', type=str,
    #                     help='input image root path')
    # parser.add_argument('-o', '--output', type=str,
    #                     help='output database path')
    # parser.add_argument('-e', '--exts-list', default=['png', 'jpg'])
    # args = parser.parse_args()

    # input_folder = '/mnt/dataset/CASIA-WebFaces/datasets' if not args.input else args.input
    # output_database = '/mnt/dataset/CASIA-WebFaces/database' if not args.output else args.output

    # print('CASIA WebFace dataset to LMDB converter')
    # print(f'    input:  {input_folder}')
    # print(f'    output: {output_database}')
    # casia_webface_to_lmdb(input_folder, output_database, limit=2000)
