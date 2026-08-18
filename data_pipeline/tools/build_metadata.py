import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import random
import argparse
random.seed(1234)
from tqdm import tqdm

from engines import LMDBEngine

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', type=str, default=None, required=True)
parser.add_argument('--reference_json', type=str, default=None)
args = parser.parse_args()

### split_metadata_lmdb
lmdb_engine = LMDBEngine(os.path.join(args.source_path, 'data_lmdb'), write=False)
all_keys = lmdb_engine.keys()
all_meta_data, all_motion_code = [], []
for key in tqdm(all_keys):
    video_length = lmdb_engine[key]['motioncode'].shape[0]
    motion_code = torch.from_numpy(lmdb_engine[key]['motioncode'])
    all_meta_data.append([key, video_length])
    all_motion_code.append(motion_code.cuda())
lmdb_engine.close()
all_motion_code = torch.cat(all_motion_code, dim=0)

random.shuffle(all_meta_data)
if args.reference_json is not None:
    reference_json = json.load(open(args.reference_json, 'r'))
    all_test_keys = reference_json['test']
    all_train_keys = [key for key in all_meta_data if key not in all_test_keys]
    all_val_keys = reference_json['val']
else:
    all_test_keys = all_meta_data[-500:]
    all_train_keys = all_meta_data[:-500]
    all_val_keys = random.sample(all_test_keys, 10)

meta_data = {'train': all_train_keys, 'test': all_test_keys, 'val': all_val_keys}
json.dump(meta_data, open(os.path.join(args.source_path, 'metadata.json'), 'w'))


print("All motion shape: {}".format(all_motion_code.shape))
motion_mean = all_motion_code.mean(dim=0)
motion_std = all_motion_code.std(dim=0)
print("Motion mean:")
print(motion_mean)
print("Motion std:")
print(motion_std)
json.dump({
        "motion_mean": motion_mean.cpu().numpy().tolist(), 
        "motion_std": motion_std.cpu().numpy().tolist()
    }, open(f'{args.source_path}/metadata_stats.json', 'w')
)
