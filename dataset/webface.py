#!/usr/bin/env python3
import lmdb
import pickle
import numpy as np

class WebfaceDataset:
    def __init__(self, env, keys=None, classes=None):
        self._env = env
        
        if keys is None:
            self._keys = self._reload_keys()
        else:
            self._keys = keys
        self._index = self._build_index(self._keys)
        
        if classes is None:
            self._classes = list(self._index.keys())
        else:
            self._classes = classes
    
    @classmethod
    def open(cls, database):
        return cls(env = lmdb.open(database, map_size=1099511627776))

    @property
    def env(self):
        return self._env

    @property
    def keys(self):
        return self._keys

    @property
    def index(self):
        return self._index
    
    @property
    def classes(self):
        return self._classes

    def _build_index(self, keys):
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

        
    def _reload_keys(self):
        with self._env.begin(write=False) as txn:
            # build index
            keys = list(txn.cursor().iternext(values=False))
            keys = [x.decode('utf-8') for x in keys if x[0] != ord('_')]
        return keys

    def __repr__(self):
        return f'Webface dataset keys={len(self.keys)} index={len(self.index)} classes={len(self.classes)}'

if __name__ == "__main__":
    ds = WebfaceDataset.open('/mnt/dataset/CASIA-WebFaces/database')
    print(ds)

    # split to different datasets
    val_set_size = int(len(ds.keys) * 0.1)
    train_set_size = len(ds.keys) - val_set_size
    
    shuffled_keys = ds.keys.copy()
    np.random.shuffle(shuffled_keys)
    val_keys = shuffled_keys[:val_set_size]
    train_keys = shuffled_keys[val_set_size:]
    
    val_ds = WebfaceDataset(ds.env, val_keys, ds.classes)
    train_ds = WebfaceDataset(ds.env, train_keys, ds.classes)
    
    print(val_ds)
    print(train_ds)
