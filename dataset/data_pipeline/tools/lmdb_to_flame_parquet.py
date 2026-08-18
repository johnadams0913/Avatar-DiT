#!/usr/bin/env python
import argparse
import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from engines import LMDBEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', '-l', required=True)
    parser.add_argument('--output_dir', '-o', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lmdb_engine = LMDBEngine(args.lmdb_path, write=False)
    keys = lmdb_engine.keys()
    print(f'Found {len(keys)} entries in LMDB')

    for key in tqdm(keys):
        data = lmdb_engine[key]
        motion = np.asarray(data['motioncode']).astype(np.float32)
        cols = [f'm_{i}' for i in range(motion.shape[1])]
        df = pd.DataFrame(motion, columns=cols)
        df.to_parquet(osp.join(args.output_dir, f'{key}.parquet'))

    lmdb_engine.close()
    print('Done')


if __name__ == '__main__':
    main()
