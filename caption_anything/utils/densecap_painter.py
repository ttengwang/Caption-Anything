import cv2
import json
import numpy as np
from typing import List
import random
from typing import Union

def draw_bbox(img: Union[np.ndarray, str], save_name: str, bbox: List[dict], show_caption: bool = False):
    """
        bbox: [{'image_id': str, 'bbox': [x1, y1, x2, y2], 'caption': str}, ...]
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        
    RGB = [0, 50, 100, 150, 200, 250]
    for box in bbox:
        box['bbox'] = [int(_) for _ in box['bbox']]
        x1, y1, x2, y2 = box['bbox']
        caption = box['caption']
        box_color = random.choices(RGB, k = 3)
        (text_width, text_height), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, thickness = 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color = box_color, thickness = 2)
        if show_caption:
            cv2.putText(img, caption, (x1, y1 + text_height), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = box_color, thickness = 2)

    cv2.imwrite(save_name, img)
    # cv2.imshow('visualise', img)
    # cv2.waitKey(0)

def parse_bbox(anno, image_id: int = None):

    with open(anno, 'r') as f:
        predictions = json.load(f)
    
    if image_id is None:
        image_id = next(iter(predictions))
        
    return predictions[image_id]
    
def gt_bbox(anno, img_name: int = None):

    with open(anno, 'r') as f:
        annotations = json.load(f)
    annotations = annotations['annotations']

    gt = []
    img_name = int(img_name[:-4])
    for annotation in annotations:
        if annotation['image_id'] == 63:
            x1, y1, w, h = annotation['bbox']
            gt.append({'bbox': [x1, y1, x1 + w, y1 + h], 'caption': annotation['caption']})
    return gt

if __name__ == '__main__':

    img_name = '63.jpg'
    show_caption = True
    anno = 'vg_dense_captioning_blip2_top48_0.88_1000_0.96_debugTrue_predictions_shard_all.json'

    img = cv2.imread(img_name)
    examp_bbox = parse_bbox(anno)
    ground_truth_bbox = gt_bbox('test.json', img_name)
    draw_bbox(img, 'GT.jpg', ground_truth_bbox, show_caption)
    draw_bbox(img, 'Pred.jpg', examp_bbox, show_caption)