#!/usr/bin/env python3
import os
import lmdb
import pickle
import numpy as np

def keys_to_dict(keys):
    index = {}
    for key in keys:
        # skip internal records
        if key.startswith('_'):
            continue
        # skip invalid keys
        parts = key.split('/')
        if len(parts) != 2:
            continue
        if index.get(parts[0]) is None:
            index[parts[0]] = []
        index[parts[0]].append(parts[1])
    return index

def dump_lmdb(database: str):
    env = lmdb.open(database, map_size=1099511627776)
    with env.begin(write=False) as txn:
        # build index
        keys = list(txn.cursor().iternext(values=False))
        keys = [x.decode('utf-8') for x in keys if x[0] != ord('_')]
        # keys = list(map(lambda x: x.decode('utf-8'), keys))
        # keys = list(filter(lambda x: not x.startswith('_'), keys))

        # split to different datasets
        val_set_size = int(len(keys) * 0.1)
        # train_set_size = len(keys) - val_set_size
        
        # shuffled_keys = keys.copy()
        np.random.shuffle(keys)
        val_set = keys[:val_set_size]
        train_set = keys[val_set_size:]

        val_index = keys_to_dict(val_set)
        train_index = keys_to_dict(train_set)
        
        # index = pickle.loads(txn.get('_index'.encode('utf-8')))
        # total = pickle.loads(txn.get('_total'.encode('utf-8')))
        print(f'val classes classes={len(val_index)} records={len(val_set)}')
        print(f'train classes classes={len(train_index)} records={len(train_set)}')

        # for k in index:
        #     print(f'class: {k} files: {len(index[k])}')
        #     key = f'{k}/{index[k][0]}'
        #     data = pickle.loads(txn.get(key.encode('utf-8')))
        #     print(f' image shape: {data.image.shape}')
            # image0 = pickle.loads(txn[])
            # r = txn[i].decode('ascii').split('/')
            # if keys.get(r[0]) is None:
            #     keys[r[0]] = []
            # keys[r[0]] = keys[r[0]] + r[1:]
        # print(keys['0000045'])
        # v = np.array(keys['0000045'])
        # rnd = np.random.choice(range(0, len(v)), 1)
        # print(v[rnd])
        # rnd3 = np.random.choice(range(0, len(v)), 3)
        # print(v[rnd3])

if __name__ == "__main__":
    dump_lmdb("/mnt/dataset/CASIA-WebFaces/database")