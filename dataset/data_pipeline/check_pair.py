import os
import os.path as osp
import json
from tqdm import tqdm

body_path = "./seamless_smpl"
face_path = "data_lmdb/seamless_interaction_pairs_lite.json"

body_files = os.listdir(body_path)
face_dict = json.load(open(face_path, "r"))
face_files = [osp.basename(file) for key in face_dict for file in face_dict[key]]

union_file = []
for file in tqdm(body_files):
    if file in face_files:
        union_file.append(file)
    else:
        print(f"{file} is not in face smpl checklist.")

print(f"Total files: {len(body_files)}, covered files: {len(union_file)}")
json.dump(union_file, open("union_files.json", "w"))