#!/usr/bin/env python
# Copyright (c) Ruicong Liu and Xuangeng Chu (xg.chu@outlook.com)

import os
import math
import torch
import pickle
import torchvision
from scipy import signal
from tqdm.rich import tqdm
import python_speech_features

from .SyncNet import SyncNetModel
from .utils import read_all_video_frames, read_audio_samples

class SyncNetEngine:
    def __init__(self, output_dir, device='cuda'):
        # Video 25 FPS, Audio 16000HZ
        self.device = device
        self.vshift = 15
        self.batch_size = 20
        self.output_dir = os.path.join(output_dir, 'sync_data')
        os.makedirs(self.output_dir, exist_ok=True)
        # model
        self.syncnet_model = SyncNetModel()
        self.syncnet_model.load_state_dict(torch.load('./assets/syncnet_v2.model', map_location='cpu'))
        self.syncnet_model.to(device)
        self.syncnet_model.eval()

    @torch.inference_mode()
    def evaluate(self, video_path):
        images, fps = read_all_video_frames(video_path, target_shape=(224, 224), device=self.device)
        imtv = images[:, [2, 1, 0]].permute(1, 0, 2, 3)[None, :, :, :, :]

        audio, sample_rate = read_audio_samples(video_path)
        audio = audio * 32768.0
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        cct = torch.stack([torch.tensor(i) for i in mfcc])[None, None].float().to(self.device)

        min_length = min(len(images), math.floor(len(audio)/640))
        
        lastframe = min_length-5
        im_feat, cc_feat = [], []
        for i in tqdm(range(0, lastframe, self.batch_size), desc="Running SyncNet"):
            im_batch = [imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+self.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.syncnet_model.forward_lip(im_in)
            im_feat.append(im_out)

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+self.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.syncnet_model.forward_aud(cc_in)
            cc_feat.append(cc_out)
        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)
        dists = calc_pdist(im_feat,cc_feat,vshift=self.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = self.vshift - minidx
        conf   = torch.median(mdist) - minval

        fdist   = torch.stack([dist[minidx] for dist in dists])
        fconf   = torch.median(mdist) - fdist
        fconfm  = signal.medfilt(fconf.cpu().numpy(), kernel_size=9)

        # dump piclke
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(self.output_dir, f'{video_name}.pkl')
        results = {'video_name': video_name, 'offset': offset.item(), 'conf': conf.item(), 'fconfm': fconfm, 'dists': torch.stack(dists).cpu().numpy()}
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        return results


def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists
