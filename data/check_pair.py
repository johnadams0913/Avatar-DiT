import os
import argparse
import threading
import os.path as osp

from glob import glob


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', required=True, help='dataroot path')
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--target', '-t', type=str, default='color')
    return parser.parse_args()

def processing(thread_id, img_files, target_key, pair_key):
    data_size = len(img_files)

    for i in range(data_size):
        if not (osp.exists(img_files[i])
                and osp.exists(img_files[i].replace(target_key, pair_key[0]))
                and osp.exists(img_files[i].replace(target_key, pair_key[1]))):
            print(f"{img_files[i]} does not have a pair image in {pair_key}, deleted")
            os.remove(img_files[i])
        if i % 5000 == 0:
            print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))


def create_threads(opt):
    inputs_keys = ["densepose", "images", "masks"]
    dataroot = osp.abspath(opt.dataroot)
    inputs_keys.remove(opt.target)

    dirs = [osp.join(opt.dataroot, d) for d in os.listdir(dataroot)]
    img_files = [
        file for ext in IMAGE_EXTENSIONS for d in dirs for file in
        glob(osp.join(d, osp.join(opt.target, f"*.{ext}")))
    ]
    data_size = len(img_files)
    print('total data size: {}'.format(data_size))
    num_threads = opt.num_threads

    if num_threads == 0:
        processing(0, img_files, opt.target, inputs_keys)
    else:
        thread_size = data_size // num_threads
        threads = []
        for t in range(num_threads):
            if t == num_threads - 1:
                thread = threading.Thread(
                    target=processing,
                    args=(t, img_files[t*thread_size: ], opt.target, inputs_keys)
                )
            else:
                thread = threading.Thread(
                    target=processing,
                    args=(t, img_files[t*thread_size: (t+1)*thread_size], opt.target, inputs_keys)
                )
            threads.append(thread)
        for t in threads:
            t.start()
        thread.join()


if __name__ == '__main__':
    opt = get_options()
    create_threads(opt)
