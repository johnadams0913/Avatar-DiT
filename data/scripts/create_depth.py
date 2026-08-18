import os
import zipfile
import numpy as np
import os.path as osp
import PIL.Image as Image

from tqdm import tqdm
from glob import glob
# from ckpt_util import load_weights
# from preprocessor.anime_segment import ISNetDIS
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt

dataroot = "./datasets/fashionvideo/frame"
save_path = "./datasets/fashionvideo/depth"

bic = transforms.InterpolationMode.BICUBIC
IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
colormap = plt.get_cmap('inferno')

def save_masks(save_path, masks, original_sizes, basenames, cut_sizes, points):
    masks = (masks.clamp(-1, 1) * 255.).squeeze(1).cpu().numpy().astype(np.uint8)

    for mask, size, bn, cts, pts in zip(masks, original_sizes, basenames, cut_sizes, points):
        dirname = osp.join(save_path, osp.dirname(bn))
        if not osp.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        mask = mask[pts[0]: pts[0]+cts[0], pts[1]: pts[1]+cts[1]]
        mask = Image.fromarray(mask, mode='L').resize((int(size[1]), int(size[0])), Image.Resampling.BICUBIC)
        mask.save(osp.join(dirname, osp.basename(bn)))

def save_depths(save_path, depths: torch.Tensor, original_sizes, basenames):
    b = depths.shape[0]
    dmin = depths.reshape(b, -1).min(dim=1)[0].reshape(b, 1, 1)
    dmax = depths.reshape(b, -1).max(dim=1)[0].reshape(b, 1, 1)

    depths = (depths - dmin) / (dmax - dmin) * 255.0
    depths = depths.clamp(0, 255).cpu().numpy().astype(np.uint8)
    #depths = (depths.clamp(-1, 1) * 255.).cpu().numpy().astype(np.uint8)

    for depth, size, bn in zip(depths, original_sizes, basenames):
        dirname = osp.join(save_path, osp.dirname(bn))
        if not osp.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        # depth = depth[pts[0]: pts[0]+cts[0], pts[1]: pts[1]+cts[1]]
        # depth = Image.f7romarray(depth, mode='L')
        depth = Image.fromarray(depth, mode='L').resize((int(size[1]), int(size[0])), Image.Resampling.BICUBIC)
#        depth = colormap(depth)
 #       depth = Image.fromarray(depth, mode='RGB')
        depth.save(osp.join(dirname, osp.basename(bn)))



def resize_with_ratio(img, new_size):
    """ This function resizes the longer edge to new_size, instead of the shorter one in PyTorch """
    w, h = img.size
    if w > h:
        img = transforms.Resize((int(h / w * new_size), new_size), bic)(img)
    else:
        img = transforms.Resize((new_size, int(w / h * new_size)), bic)(img)
    return img


class ZipDataset(data.Dataset):
    def __init__(self, dataroot, image_size, image_processor):
        super().__init__()
        self.zip_path = osp.abspath(dataroot)
        self.zip_reader = None
        self.image_processor = image_processor

        self.image_files = sorted([
            file for zip in glob(osp.join(self.zip_path, "*.zip"))
            for file in zipfile.ZipFile(zip).namelist() if
            (not file.endswith("/") and file.split(".")[-1] in IMAGE_EXTENSIONS)
        ])
        self.data_size = len(self)
        self.image_size = image_size

    def fresh_zip_files(self, zipid):
        self.zip_reader = zipfile.ZipFile(osp.join(self.zip_path, f"{zipid}.zip"), "r")

    def get_images(self, index):
        filename = self.image_files[index]
        self.fresh_zip_files(osp.dirname(filename))

        img = Image.open(self.zip_reader.open(filename, "r")).convert("RGB")
        width, height = img.size

        # img = resize_with_ratio(img, self.image_size)
        #
        # padding = (0, 0, 0)
        # nw, nh = img.size
        # square_image = Image.new('RGB', (self.image_size, self.image_size), padding)
        # left = (self.image_size - nw) // 2
        # top = (self.image_size - nh) // 2
        #
        # square_image.paste(img, (int(left), int(top)))
        img = self.image_processor(img, return_tensors='pt').pixel_values
        img = img.squeeze(0)

        return img, (height, width)


    def __getitem__(self, index):
        img, original_size = self.get_images(index)

        return {
            "image": img,
            "size": torch.Tensor(original_size),
            "filename": self.image_files[index],
        }

    def __len__(self):
        return len(self.image_files)




if __name__ == '__main__':
    
    image_size = 1024
    batch_size = 32
    num_threads = 8

    # model = ISNetDIS().eval()
    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf", use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

    # model.load_state_dict(load_weights(ckpt_path))
    model = model.cuda()
    dataset = ZipDataset(dataroot, image_size, image_processor)
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2 if num_threads > 0 else None,
    )

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            img = data["image"]
            original_size = data["size"]
            bn = data["filename"]
            # cut_size = data["cut_size"]
            # pts = data["points"]

            img = img.cuda()

            predicted = model(img)
            original_size = original_size.numpy().astype(np.int32)
            # original_size, cut_size, pts = map(lambda t: t.numpy().astype(np.int32), (original_size, cut_size, pts))
            save_depths(save_path, predicted.predicted_depth, original_size, bn)
