"""
get semantic segmentation annotations from coco data set.
"""
import shutil
from PIL import Image
import imgviz
import argparse
import os
import numpy as np
import tqdm
from pycocotools.coco import COCO

 
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)
 
 
def main(args):
    annotation_file = os.path.join(args.input_dir, 'annotations', 'instances_{}.json'.format(args.split))
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass_plant'), exist_ok=True)
    # os.makedirs(os.path.join(args.input_dir, 'JPEGImages_onlyhuman'), exist_ok=True)
    coco = COCO(annotation_file)
    # catIds = coco.getCatIds()
    # imgIds = coco.getImgIds()
    # select specific catrgory
    catIds = coco.getCatIds(catNms=['potted plant'])
    imgIds = coco.getImgIds(catIds=catIds)
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            # img_origin_path = os.path.join(args.input_dir, args.split, img['file_name'])
            # img_output_path = os.path.join(args.input_dir, 'JPEGImages_onlyhuman', img['file_name'])
            seg_output_path = os.path.join(args.input_dir, 'SegmentationClass_plant', img['file_name'].replace('.jpg', '.png'))
            # shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(mask, seg_output_path)
 
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../dataset/coco2017", type=str,
                        help="coco dataset directory")
    parser.add_argument("--split", default="train2017", type=str,
                        help="train2017 or val2017")
    return parser.parse_args()
 
 
if __name__ == '__main__':
    args = get_args()
    main(args)