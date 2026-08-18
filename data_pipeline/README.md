# Enviroment
```pip install scipy tqdm rich av torchmetrics pytubefix lmdb ffmpeg-python python_speech_features opencv-python timm face_alignment onnx2torch```
```pip install "git+https://github.com/facebookresearch/pytorch3d.git"```

# Prepare assets
Download ```FaceBoxes.pth``` from release or ```https://drive.google.com/file/d/128m1QasIwQRkrY-Eb5Epi-ShXnrZWUCQ/view```.
Download ```syncnet_v2.model``` from release or ```http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model```.

# Stage One
## Run download
```python run_download.py -v ./data_list/video_ids.txt -o ./talkinghead1kh_raw```

## Run track and split
```python run_tracksplit.py -v ./talkinghead1kh_raw -o ./clip_videos --split_id 0 --split_total 4```

## Run SyncNet
```python run_syncnet.py -v ./clip_videos/videos -o ./clip_videos --split_id 0 --split_total 4```

Now we should get a `clip_videos` folder with `videos`, `meta_data` and `sync_data`.

# Stage Two
## Run EMICA Tracking

## Run EMICA Optimization

## Run Gaze Tracking
