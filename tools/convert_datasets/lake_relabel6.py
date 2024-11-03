import argparse
import os.path as osp
import numpy as np
import mmcv
from PIL import Image

rellis_dir = "./data/lake/"
annotation_folder = "annotation/"

IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Groups = [5, 0, 0, 5, 1, 4, 5, 5, 5]

ID_seq = {}
ID_group = {}
for n, label in enumerate(IDs):
    ID_seq[label] = n
    ID_group[label] = Groups[n]

CLASSES = ("leaf", "sky", "low grass", "telephone pole", "grass", "water",
           "telephone pole wire", "bush", "tree trunk")

PALETTE = [[255, 255, 0], [127, 127, 127], [97, 127, 56],
           [255, 153, 153], [0, 255, 0], [33, 112, 178],
           [61, 59, 112], [0, 0, 255], [255, 0, 255]]

# 0 -- Background: void, sky, sign
# 1 -- Level1 (smooth) - Navigable: concrete, asphalt
# 2 -- Level2 (rough) - Navigable: gravel, grass, dirt, sand, mulch
# 3 -- Level3 (bumpy) - Navigable: Rock, Rock-bed
# 4 -- Non-Navigable (forbidden) - water
# 5 -- Obstacle - tree, pole, vehicle, container/generic-object, building, log, 
#                 bicycle(could be removed), person, fence, bush, picnic-table, bridge,


def raw_to_seq(seg):
    # Check if the image is 3D and reduce to 2D if so
    if seg.ndim == 3:
        seg = seg[:, :, 0]  # Use only the first channel if it's multi-channel

    h, w = seg.shape
    out1 = np.zeros((h, w))
    out2 = np.zeros((h, w))
    for i in IDs:
        out1[seg == i] = ID_seq[i]
        out2[seg == i] = ID_group[i]

    return out1, out2


# Processing function for train, val, and test splits
def process_split(split_name):
    with open(osp.join(rellis_dir, f'{split_name}.txt'), 'r') as r:
        i = 0
        for l in r:
            print(f"{split_name}: {i}")
            file_client_args = dict(backend='disk')
            file_client = mmcv.FileClient(**file_client_args)
            img_bytes = file_client.get(rellis_dir + annotation_folder + l.strip() + '.png')
            gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='pillow').astype(np.uint8)

            # Convert to single channel and process
            out1, out2 = raw_to_seq(gt_semantic_seg)

            # Save the outputs
            mmcv.imwrite(out1, rellis_dir + annotation_folder + l.strip() + "_orig.png")
            mmcv.imwrite(out2, rellis_dir + annotation_folder + l.strip() + "_group6.png")

            i += 1


# Run processing for each split
for split in ['train', 'val', 'test']:
    process_split(split)

print("successful")
