import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

from engines import LMDBEngine

### merge_lmdb
target_path = '../d/UniTalkData_v0/data_lmdb'
source_lmdb_paths = [
    '../d/tracked/CelebV_Tracked', 
    '../d/tracked/TalkingHead1KH_Tracked/data_lmdb', 
    '../d/tracked/MultiTalk_Chinese_Tracked',
    '../d/tracked/MultiTalk_English_Tracked',
    '../d/tracked/MultiTalk_Japanese_Tracked',
    '../d/tracked/TEDTalk_Tracked',
    '../d/tracked/VFHQ_Tracked',
]
os.makedirs(target_path, exist_ok=True)

lmdb_engines = [LMDBEngine(lmdb_path, write=False) for lmdb_path in source_lmdb_paths]
lmdb_engines_names = [os.path.basename(lmdb_path).split('_')[0] for lmdb_path in source_lmdb_paths]
lmdb_engine = LMDBEngine(target_path, write=True)
for this_lmdb_engine, this_lmdb_engine_name in zip(lmdb_engines, lmdb_engines_names):
    keys = this_lmdb_engine.keys()
    for key in tqdm(keys):
        raw_payload = this_lmdb_engine.raw_load(key)
        lmdb_engine.raw_dump(key, raw_payload)
    this_lmdb_engine.close()
lmdb_engine.close()
for this_lmdb_engine in lmdb_engines:
    this_lmdb_engine.close()
