#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
from image_record import ImageRecord
from functools import reduce
from tqdm import tqdm

import lmdb

def casia_webface_to_lmdb(folder: str, database: str, exts=None, limit=sys.maxsize):
    """
    search image files in folder, store images with path into database
    key value format: | folder/file | ImageRecord |
    """
    if exts is None:
        exts = ['jpg', 'png']

    while folder.endswith('/'):
        folder = folder[:len(folder) - 1]

    index = {}

    env = lmdb.open(database, map_size=1099511627776)
    with env.begin(write=True) as txn:
        with tqdm(total=1) as pbar:
            for root, _, files in os.walk(folder):
                pbar.total += len(files)
                pbar.refresh()
                # print(f'root={root} basename={os.path.basename(root)}')
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
                    
                    key = f'{label}/{name}'
                    full_path = os.path.join(root, name)
                    # print(f'path={full_path} key={key} label={label}')
                    rec = ImageRecord.from_image(label, full_path)
                    txn.put(key.encode('utf-8'), rec.dumps())

                    pbar.update(1)

                    if pbar.n >= limit:
                        break

                if pbar.n >= limit:
                    break
            total = pbar.n
        # txn.put('_index'.encode('utf-8'), pickle.dumps(index))
        # txn.put('_total'.encode('utf-8'), pickle.dumps(total))
    # txn.commit()
    print('done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CASIA WebFace dataset to LMDB converter')
    parser.add_argument('-i', '--input', type=str,
                        help='input image folder path')
    parser.add_argument('-o', '--output', type=str,
                        help='output database path')
    parser.add_argument('-e', '--exts-list', default=['png', 'jpg'])
    args = parser.parse_args()

    input_folder = '/mnt/dataset/CASIA-WebFaces/datasets' if not args.input else args.input
    output_database = '/mnt/dataset/CASIA-WebFaces/database' if not args.output else args.output

    print('CASIA WebFace dataset to LMDB converter')
    print(f'    input:  {input_folder}')
    print(f'    output: {output_database}')
    casia_webface_to_lmdb(input_folder, output_database, limit=2000)
